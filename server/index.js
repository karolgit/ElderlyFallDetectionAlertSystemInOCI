import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import axios from 'axios';
import multer from 'multer';
import FormData from 'form-data';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import * as common from 'oci-common';
import * as objectstorage from 'oci-objectstorage';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000';
const OCI_REGION = process.env.OCI_REGION || 'us-chicago-1';
const INPUT_BUCKET = process.env.INPUT_BUCKET || 'guidanceaiwatch_input';

let osClient = null;
let osNamespace = null;

async function ensureObjectStorage() {
  if (osClient && osNamespace) return;
  let provider;
  try {
    if (process.env.OCI_RESOURCE_PRINCIPAL_VERSION) {
      provider = new common.ResourcePrincipalAuthenticationDetailsProvider();
    } else {
      provider = new common.InstancePrincipalsAuthenticationDetailsProvider();
    }
  } catch (e) {
    provider = new common.ConfigFileAuthenticationDetailsProvider();
  }
  osClient = new objectstorage.ObjectStorageClient({ authenticationDetailsProvider: provider });
  if (OCI_REGION) {
    osClient.regionId = OCI_REGION;
  }
  const ns = await osClient.getNamespace({});
  osNamespace = ns.value || ns.data || (ns?.namespace || '').toString();
}

function makeObjectName(original) {
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const safe = (original || 'video.mp4').replace(/[^a-zA-Z0-9._-]/g, '_');
  return `${ts}_${safe}`;
}

async function uploadBufferToBucket(bucketName, objectName, buffer, contentType) {
  await ensureObjectStorage();
  const putReq = {
    namespaceName: osNamespace,
    bucketName,
    objectName,
    contentType: contentType || 'application/octet-stream',
    putObjectBody: Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer),
  };
  await osClient.putObject(putReq);
}

app.use(cors());
app.use(morgan('dev'));
app.use(express.json({ limit: '10mb' }));

app.get('/api/health', async (req, res) => {
  try {
    const { data } = await axios.get(`${BACKEND_URL}/health`);
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: 'Backend unavailable', detail: String(err) });
  }
});

app.post('/api/analyze_frame', async (req, res) => {
  try {
    const { data } = await axios.post(`${BACKEND_URL}/analyze_frame`, req.body, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 60000,
    });
    res.json(data);
  } catch (err) {
    if (axios.isAxiosError(err) && err.response) {
      res.status(err.response.status).json(err.response.data);
    } else {
      res.status(502).json({ error: 'Proxy error', detail: String(err) });
    }
  }
});

const upload = multer({ storage: multer.memoryStorage() });
app.post('/api/analyze_video', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }
    const saveToBucket = String(req.body?.save_to_bucket).toLowerCase() === 'true';
    if (saveToBucket) {
      try {
        await uploadBufferToBucket(INPUT_BUCKET, makeObjectName(req.file.originalname || 'video.mp4'), req.file.buffer, req.file.mimetype);
      } catch (e) {
        // non-fatal: proceed even if upload fails
        console.warn('Input upload to bucket failed:', e?.message || e);
      }
    }
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'video.mp4',
      contentType: req.file.mimetype || 'video/mp4',
    });
    form.append('save_to_bucket', saveToBucket ? 'true' : 'false');
    const { data } = await axios.post(`${BACKEND_URL}/analyze_video`, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      timeout: 10 * 60 * 1000,
    });
    res.json(data);
  } catch (err) {
    if (axios.isAxiosError(err) && err.response) {
      res.status(err.response.status).json(err.response.data);
    } else {
      res.status(502).json({ error: 'Proxy error', detail: String(err) });
    }
  }
});

app.post('/api/annotate_video', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }
    const saveToBucket = String(req.body?.save_to_bucket).toLowerCase() === 'true';
    if (saveToBucket) {
      try {
        await uploadBufferToBucket(INPUT_BUCKET, makeObjectName(req.file.originalname || 'video.mp4'), req.file.buffer, req.file.mimetype);
      } catch (e) {
        console.warn('Input upload to bucket failed:', e?.message || e);
      }
    }
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'video.mp4',
      contentType: req.file.mimetype || 'video/mp4',
    });
    form.append('save_to_bucket', saveToBucket ? 'true' : 'false');
    const resp = await axios.post(`${BACKEND_URL}/annotate_video`, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      responseType: 'stream',
      timeout: 10 * 60 * 1000,
    });

    const dispo = resp.headers['content-disposition'];
    if (dispo) {
      res.setHeader('Content-Disposition', dispo);
    }
    res.setHeader('Content-Type', 'video/mp4');
    resp.data.pipe(res);
  } catch (err) {
    if (axios.isAxiosError(err) && err.response) {
      res.status(err.response.status).json(err.response.data);
    } else {
      res.status(502).json({ error: 'Proxy error', detail: String(err) });
    }
  }
});

// Async annotate job endpoints
app.post('/api/annotate_video_async', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No video file provided' });
    const saveToBucket = String(req.body?.save_to_bucket).toLowerCase() === 'true';
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'video.mp4',
      contentType: req.file.mimetype || 'video/mp4',
    });
    form.append('save_to_bucket', saveToBucket ? 'true' : 'false');
    const { data } = await axios.post(`${BACKEND_URL}/annotate_video_async`, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      timeout: 60 * 1000,
    });
    res.json(data);
  } catch (err) {
    if (axios.isAxiosError(err) && err.response) {
      res.status(err.response.status).json(err.response.data);
    } else {
      res.status(502).json({ error: 'Proxy error', detail: String(err) });
    }
  }
});

app.get('/api/annotate_progress/:jobId', async (req, res) => {
  try {
    const { data } = await axios.get(`${BACKEND_URL}/annotate_progress/${req.params.jobId}`);
    res.json(data);
  } catch (err) {
    if (axios.isAxiosError(err) && err.response) {
      res.status(err.response.status).json(err.response.data);
    } else {
      res.status(502).json({ error: 'Proxy error', detail: String(err) });
    }
  }
});

app.get('/api/annotate_result/:jobId', async (req, res) => {
  try {
    const resp = await axios.get(`${BACKEND_URL}/annotate_result/${req.params.jobId}`, { responseType: 'stream' });
    const dispo = resp.headers['content-disposition'];
    if (dispo) res.setHeader('Content-Disposition', dispo);
    res.setHeader('Content-Type', 'video/mp4');
    resp.data.pipe(res);
  } catch (err) {
    if (axios.isAxiosError(err) && err.response) {
      res.status(err.response.status).json(err.response.data);
    } else {
      res.status(502).json({ error: 'Proxy error', detail: String(err) });
    }
  }
});

const staticDir = path.resolve(__dirname, '../frontend/dist');
if (fs.existsSync(staticDir)) {
  app.use(express.static(staticDir));
  app.get('*', (req, res) => {
    res.sendFile(path.join(staticDir, 'index.html'));
  });
}

app.listen(PORT, () => {
  console.log(`Node proxy listening on http://localhost:${PORT} -> ${BACKEND_URL}`);
});
