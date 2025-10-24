## Elderly Fall Detection - React + Node + FastAPI + PyTorch (Sports2D-ready)

### Prereqs
- macOS Apple Silicon (MPS) or Linux/Windows with NVIDIA CUDA
- Python 3.10+
- Node.js 18+

### Backend (Python FastAPI)

1) Create venv and install deps
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Apple Silicon (MPS) notes
- PyTorch 2.x supports MPS. Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Default device will be MPS on Apple Silicon if available.

3) CUDA notes
- Install a PyTorch build with CUDA support per pytorch.org instructions.
- Set `PREFERRED_DEVICE=cuda` to force CUDA.

4) Run backend
```bash
cd backend
source .venv/bin/activate
./run.sh
```
Backend runs on `http://127.0.0.1:8000`.

### Node proxy server

```bash
cd server
npm install
npm run dev
```
Proxy runs on `http://127.0.0.1:3000` and forwards to FastAPI.

### Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```
Open `http://127.0.0.1:5173`. The dev server proxies `/api` -> Node proxy.

### Build for production
- Build frontend: `cd frontend && npm run build`
- Start Node to serve static: `cd server && npm start`

### API
- `GET /api/health`
- `POST /api/analyze_frame` body: `{ image_base64: string }`
- `POST /api/analyze_video` form-data: `file: <video>`

### Sports2D
This backend tries to use Sports2D if installed; otherwise falls back to torchvision Keypoint R-CNN. To enable Sports2D, install the package in the backend venv. The adapter is in `backend/app/pose.py`.

### Notes
- Pose overlay is drawn in the frontend based on returned keypoints.
- Fall detection uses a heuristic on torso angle and bbox aspect ratio. Replace with a trained model if desired in `backend/app/fall.py`.
