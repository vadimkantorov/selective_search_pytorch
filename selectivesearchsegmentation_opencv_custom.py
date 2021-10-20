import os
import ctypes
import torch

class SelectiveSearchOpenCVCustom(torch.nn.Module):
    def __init__(self, preset = 'fast', lib_path = 'selectivesearchsegmentation_opencv_custom.so', max_num_rects = 4096, max_num_seg = 16, max_num_bit = 64):
        super().__init__()
        self.bind = ctypes.CDLL(lib_path)
        self.bind.process.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_float
        ]
        self.bind.process.restype = ctypes.c_int
        self.bind_num_rects = ctypes.c_int()
        self.bind_num_seg = ctypes.c_int()
        self.preset = preset
        self.max_num_rects = max_num_rects
        self.max_num_seg = max_num_seg
        self.max_num_bit = max_num_bit
        self.byte_nonzero = [[i for i in range(s.bit_length()) if s & (1 << i)] for s in range(256)]
    
    def bit_nonzero(self, bits):
        return [ [i * 8 + (7 - i) for i, b in enumerate(bit) for k in self.byte_nonzero[b] ] for bit in bits ]

    @staticmethod
    def get_region_mask(reg_lab, regs):
        return torch.stack([(reg_lab[reg['plane_id'][:-1]][..., None] == torch.tensor(list(reg['ids']), device = reg_lab.device, dtype = reg_lab.dtype)).any(dim = -1) for reg in regs])

    def forward(self, img):
        img = torch.as_tensor(img).contiguous()
        assert img.is_floating_point() and img.ndim == 4 and img.shape[-3] == 3

        rects = torch.zeros(size = (img.shape[0], self.max_num_rects, 4), dtype = torch.int32)
        reg   = torch.zeros(size = (img.shape[0], self.max_num_rects, 3), dtype = torch.int32)
        bit   = torch.zeros(size = (img.shape[0], self.max_num_rects, self.max_num_bit), dtype = torch.uint8)
        seg   = torch.zeros(size = (img.shape[0], self.max_num_seg, img.shape[-2], img.shape[-1]), dtype = torch.int32)
        
        boxes_xywh, regions, reg_lab = [], [], []

        for k, img_opencv in enumerate((img.movedim(-3, -1) * 255).to(torch.uint8)):
            self.bind_num_rects.value = self.max_num_rects
            self.bind_num_seg.value = self.max_num_seg
            res = self.bind.process(
                img_opencv.data_ptr(), img_opencv.shape[0], img_opencv.shape[1], 
                rects[k].data_ptr(), ctypes.addressof(self.bind_num_rects),
                seg[k].data_ptr(), ctypes.addressof(self.bind_num_seg),
                reg[k].data_ptr(),
                bit[k].data_ptr(), bit.shape[-1],
                self.preset.encode(), 0, 0, 0.0
            )
            assert res == 0
            
            boxes_xywh.append(rects[k, :self.bind_num_rects.value])
            regions.append([dict(plane_id = (k, lab, -1), ids = b, level = lev, id = id) for (lev, lab, id), b in zip(reg[k].tolist(), self.bit_nonzero(bit[k].tolist())) for i in range(len(boxes_xywh[-1]))])
            reg_lab.append(seg[k, :self.bind_num_seg.value])

        return boxes_xywh, regions, torch.stack(reg_lab)
