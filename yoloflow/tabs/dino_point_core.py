from __future__ import annotations
import contextlib, collections, cv2, numpy as np, torch, torch.nn.functional as F, timm
from PIL import Image
from timm.data import create_transform, resolve_data_config

class DinoCore:
    def __init__(self, model="vit_small_patch8_224.dino",
                 side=336, device="cuda", stride=3, window=2, alpha=0.3):
        self.side, self.stride, self.device = side, stride, device
        self.window, self.alpha = window, alpha
        self.model = timm.create_model(model, pretrained=True,
                                       dynamic_img_size=True).eval().to(device)
        self.patch = self.model.patch_embed.patch_size[0]
        cfg = resolve_data_config({}, model=self.model)
        cfg['input_size'] = (3, side, side)
        self.tf = create_transform(**cfg, is_training=False)
        self.amp = (torch.cuda.amp.autocast if torch.cuda.is_available() and
                    device.startswith("cuda") else contextlib.nullcontext)
        self.ref_vec: torch.Tensor | None = None
        self.hist: collections.deque = collections.deque()
        self.idx = 0

    @torch.inference_mode()
    def _embed(self, bgr: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        x = self.tf(pil).unsqueeze(0).to(self.device)
        with self.amp(dtype=torch.float16):
            tok = self.model.forward_features(x)[:, 1:, :].squeeze(0)
        return F.normalize(tok.float(), dim=1)

    def _smooth(self, pts: list[tuple[int,int]]) -> list[tuple[int,int]]:
        if self.window > 0:
            self.hist.extend(pts)
            while len(self.hist) > self.window: self.hist.popleft()
            arr = np.array(self.hist).reshape(-1, len(pts), 2)
            return [tuple(map(int, arr[:, i].mean(axis=0))) for i in range(len(pts))]
        if not self.hist:
            self.hist.extend(pts); return pts
        out=[]
        for (xn,yn),(xo,yo) in zip(pts,self.hist):
            out.append((int(self.alpha*xn+(1-self.alpha)*xo),
                        int(self.alpha*yn+(1-self.alpha)*yo)))
        self.hist.clear(); self.hist.extend(out); return out

    # public API -------------------------------------------------------------
    def init_from_bbox(self, frame_bgr: np.ndarray, box):
        x,y,w,h = box
        cx, cy = x+w//2, y+h//2
        frame = cv2.resize(frame_bgr, (self.side, self.side))
        emb = self._embed(frame)
        col, row = cx * self.side // frame_bgr.shape[1], cy * self.side // frame_bgr.shape[0]
        idx = (row//self.patch)*(self.side//self.patch)+(col//self.patch)
        self.ref_vec = emb[idx]
        self.hist.clear()
        self.idx = 0

    def track(self, frame_bgr: np.ndarray) -> tuple[int,int]:
        assert self.ref_vec is not None, "call init_from_bbox first"
        frame_r = cv2.resize(frame_bgr, (self.side, self.side))
        if self.idx % self.stride == 0:
            emb = self._embed(frame_r)
            idx = int((emb @ self.ref_vec).argmax())
            cols = self.side // self.patch
            row, col = divmod(idx, cols)
            cx = col * self.patch + self.patch//2
            cy = row * self.patch + self.patch//2
            pts_raw = [(cx, cy)]
        else:
            # simple linear flow (one point)
            flow = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(self.prev_small, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY),
                None, 0.5,3,15,3,5,1.2,0)
            cx, cy = self.prev_pt
            fx, fy = flow[cy, cx]
            pts_raw = [(int(cx+fx), int(cy+fy))]
        self.prev_small = frame_r
        self.prev_pt = pts_raw[0]
        self.idx += 1
        return self._smooth(pts_raw)[0]