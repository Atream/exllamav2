import torch
from typing import Dict

class CUDAGraphRunner:

    def __init__(self):
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        model,
        hidden_states,
        past_len,
        batch_size,
        cache,
        input_mask,
        position_offsets,
        main_device,
        **kwargs,
    ) -> None:
        assert self.graph is None
        # Capture the graph.
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        #self.graph.enable_debug_mode()
        
        self.batch_size = batch_size
        past_len_tensor = torch.tensor([past_len] * self.batch_size, dtype = torch.int, pin_memory=True)
        
        # torch.cuda.set_device can't set "cuda", must have a index
        if main_device == "cuda":
            main_device = "cuda:0"
        torch.cuda.set_device(main_device)
        self.main_device = main_device
        capture_stream = torch.cuda.Stream()
        
        with torch.cuda.graph(self.graph, stream = capture_stream):
            graph_event = torch.cuda.Event()
            capture_stream.record_event(graph_event)
            for dev in model.tp_context.all_devs:
                model.get_device_context(dev).stream.wait_event(graph_event)
            logits=model.forward_chunk(hidden_states = hidden_states,
                                past_len_tensor = past_len_tensor,
                                cache = cache,
                                input_mask = input_mask,
                                position_offsets = position_offsets,
                                use_cuda_graph = False,
                                is_capturing = True,
                                **kwargs)
            for dev in model.tp_context.all_devs:
                dev_end_event = torch.cuda.Event()
                model.get_device_context(dev).stream.record_event(dev_end_event)
                capture_stream.wait_event(dev_end_event)
                
            torch.cuda.set_device(main_device)
            torch.cuda.set_stream(capture_stream)
        torch.cuda.synchronize(self.main_device)
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        #self.graph.debug_dump("cuda_graph_hooked.dot")

        # Save the input and output buffers.
        self.input_buffers = {
            "hidden_states": hidden_states,
            "input_mask": input_mask,
            "position_offsets": position_offsets,
            "past_len_tensor": past_len_tensor,
        }
        self.output_buffers = {"logits": logits}
        return

    def forward(
        self,
        hidden_states,
        past_len,
        input_mask,
        position_offsets,
    ) -> torch.Tensor:
        # Copy the input tensors to the input buffers.
        pl = torch.tensor([past_len] * self.batch_size, dtype = torch.int)
        self.input_buffers["hidden_states"].copy_(hidden_states)
        if self.input_buffers["input_mask"] is not None:
            self.input_buffers["input_mask"].copy_(input_mask)
        if self.input_buffers["position_offsets"] is not None:
            self.input_buffers["position_offsets"].copy_(position_offsets)
        self.input_buffers["past_len_tensor"].copy_(pl)

        # Run the graph.
        #print("begin replay")
        #time.sleep(1)
        self.graph.replay()
        torch.cuda.synchronize(self.main_device)
        # Return the output tensor.
        return self.output_buffers["logits"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)