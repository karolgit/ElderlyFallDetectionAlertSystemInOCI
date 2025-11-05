import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'

const API_BASE = ((import.meta as any).env?.BASE_URL ?? '/') + 'api'

type Keypoint = { x: number; y: number; score: number; name?: string }
interface PersonPose { keypoints: Keypoint[]; score: number; bbox?: number[] }

interface FrameAnalyzeResponse {
  device: { type: string; name?: string }
  people: PersonPose[]
  is_fall: boolean
  fall_score: number
}

function drawOverlay(canvas: HTMLCanvasElement, people: PersonPose[]) {
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.lineWidth = 2
  ctx.strokeStyle = '#00FF88'
  ctx.fillStyle = '#00FF88'

  const pairs: [string, string][] = [
    ['left_shoulder', 'right_shoulder'],
    ['left_hip', 'right_hip'],
    ['left_shoulder', 'left_hip'],
    ['right_shoulder', 'right_hip'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
  ]

  for (const person of people) {
    const byName = new Map(person.keypoints.map(k => [k.name ?? '', k]))
    for (const kp of person.keypoints) {
      if (kp.score < 0.3) continue
      ctx.beginPath()
      ctx.arc(kp.x, kp.y, 3, 0, Math.PI * 2)
      ctx.fill()
    }
    for (const [a, b] of pairs) {
      const ka = byName.get(a)
      const kb = byName.get(b)
      if (!ka || !kb) continue
      if (ka.score < 0.3 || kb.score < 0.3) continue
      ctx.beginPath()
      ctx.moveTo(ka.x, ka.y)
      ctx.lineTo(kb.x, kb.y)
      ctx.stroke()
    }
    if (person.bbox && person.bbox.length === 4) {
      const [x1, y1, x2, y2] = person.bbox
      ctx.strokeStyle = '#ffaa00'
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      ctx.strokeStyle = '#00FF88'
    }
  }
}

function Spinner() {
  const size = 36
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.35)', zIndex: 10 }}>
      <div style={{ width: size, height: size, border: '4px solid #fff', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
      <style>
        {`@keyframes spin { from { transform: rotate(0deg) } to { transform: rotate(360deg) } }`}
      </style>
    </div>
  )
}

function Banner({ type, message }: { type: 'success' | 'error' | 'info'; message: string }) {
  const bg = type === 'success' ? '#e7f8ef' : type === 'error' ? '#fde8e8' : '#eef2ff'
  const color = type === 'success' ? '#05603a' : type === 'error' ? '#b42318' : '#3538cd'
  return (
    <div style={{ background: bg, color, padding: '8px 12px', borderRadius: 8, border: `1px solid ${color}33` }}>
      {message}
    </div>
  )
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [livePose, setLivePose] = useState(false)
  const [webcamActive, setWebcamActive] = useState(false)
  const [lastResult, setLastResult] = useState<FrameAnalyzeResponse | null>(null)
  const [lastVideoSummary, setLastVideoSummary] = useState<string>('')
  const [lastRawJson, setLastRawJson] = useState<string>('')
  const [banner, setBanner] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null)
  const [device, setDevice] = useState<string>('')
  const [annotateProgress, setAnnotateProgress] = useState<{ jobId: string | null; percent: number | null; processed: number; total: number | null}>({ jobId: null, percent: null, processed: 0, total: null })
  const [saveToBucket, setSaveToBucket] = useState<boolean>(false)
  const liveTimer = useRef<number | null>(null)
  const pollTimer = useRef<number | null>(null)
  const liveInflight = useRef<boolean>(false)

  useEffect(() => {
    axios.get(`${API_BASE}/health`).then(r => {
      const d = r.data?.device
      setDevice(`${d?.type ?? ''} ${d?.name ? '(' + d.name + ')' : ''}`.trim())
    }).catch(() => setDevice('unknown'))
  }, [])

  useEffect(() => {
    const v = videoRef.current
    const c = overlayRef.current
    if (!v || !c) return
    const syncSizes = () => {
      if (v.videoWidth && v.videoHeight) {
        c.width = v.videoWidth
        c.height = v.videoHeight
      }
    }
    v.addEventListener('loadedmetadata', syncSizes)
    syncSizes()
    return () => {
      v.removeEventListener('loadedmetadata', syncSizes)
    }
  }, [webcamActive])

  useEffect(() => {
    if (!livePose) {
      if (liveTimer.current) {
        window.clearInterval(liveTimer.current)
        liveTimer.current = null
      }
      return
    }
    liveTimer.current = window.setInterval(() => {
      if (liveInflight.current) return
      liveInflight.current = true
      captureAndAnalyze(true).finally(() => {
        liveInflight.current = false
      })
    }, 333)
    return () => {
      if (liveTimer.current) {
        window.clearInterval(liveTimer.current)
        liveTimer.current = null
      }
    }
  }, [livePose])

  async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
    if (!videoRef.current) return
    videoRef.current.srcObject = stream
    await videoRef.current.play()
    const v = videoRef.current
    const c = overlayRef.current!
    c.width = v.videoWidth
    c.height = v.videoHeight
    setWebcamActive(true)
  }

  function stopWebcam() {
    setLivePose(false)
    if (liveTimer.current) {
      window.clearInterval(liveTimer.current)
      liveTimer.current = null
    }
    const v = videoRef.current
    if (v && v.srcObject) {
      const stream = v.srcObject as MediaStream
      stream.getTracks().forEach(t => t.stop())
      v.pause()
      v.srcObject = null
    }
    const c = overlayRef.current
    if (c) {
      const ctx = c.getContext('2d')
      if (ctx) ctx.clearRect(0, 0, c.width, c.height)
    }
    setWebcamActive(false)
    setBanner({ type: 'info', message: 'Webcam stopped.' })
  }

  async function captureAndAnalyze(silent = false) {
    if (!videoRef.current) return
    const v = videoRef.current
    const tmp = document.createElement('canvas')
    tmp.width = v.videoWidth
    tmp.height = v.videoHeight
    const ctx = tmp.getContext('2d')!
    ctx.drawImage(v, 0, 0)
    const dataUrl = tmp.toDataURL('image/jpeg')

    if (!silent) {
      setAnalyzing(true)
      setBanner({ type: 'info', message: 'Analyzing frame…' })
      setLastVideoSummary('')
    }
    try {
      const { data } = await axios.post<FrameAnalyzeResponse>(`${API_BASE}/analyze_frame`, {
        image_base64: dataUrl
      })
      setLastResult(data)
      if (!silent) setLastRawJson(JSON.stringify(data, null, 2))
      if (overlayRef.current) {
        drawOverlay(overlayRef.current, data.people)
      }
      if (!silent) setBanner({ type: 'success', message: `Frame analyzed. Fall: ${data.is_fall ? 'YES' : 'NO'} (score ${data.fall_score.toFixed(2)})` })
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Unknown error'
      if (!silent) {
        setBanner({ type: 'error', message: `Analyze failed: ${msg}` })
        setLastRawJson(JSON.stringify(err?.response?.data ?? { error: String(err) }, null, 2))
      }
    } finally {
      if (!silent) setAnalyzing(false)
    }
  }

  async function handleVideoUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    form.append('save_to_bucket', saveToBucket ? 'true' : 'false')
    setAnalyzing(true)
    setBanner({ type: 'info', message: `Uploading and analyzing video: ${file.name}` })
    setLastResult(null)
    try {
      const { data } = await axios.post(`${API_BASE}/analyze_video`, form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setLastVideoSummary(`Analyzed ${data.analyzed_frames} frames. Fall: ${data.any_fall ? 'YES' : 'NO'} (avg score ${Number(data.average_fall_score).toFixed(2)})`)
      setLastRawJson(JSON.stringify(data, null, 2))
      setBanner({ type: 'success', message: 'Video analyzed successfully.' })
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Unknown error'
      setBanner({ type: 'error', message: `Video analyze failed: ${msg}` })
      setLastRawJson(JSON.stringify(err?.response?.data ?? { error: String(err) }, null, 2))
    } finally {
      setAnalyzing(false)
      e.target.value = ''
    }
  }

  function clearAnnotatePolling() {
    if (pollTimer.current) {
      window.clearInterval(pollTimer.current)
      pollTimer.current = null
    }
  }

  async function handleAnnotateVideo(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    form.append('save_to_bucket', saveToBucket ? 'true' : 'false')
    setAnalyzing(true)
    setBanner({ type: 'info', message: `Annotating video (skeleton overlay): ${file.name}` })
    setAnnotateProgress({ jobId: null, percent: 0, processed: 0, total: null })
    try {
      const start = await axios.post(`${API_BASE}/annotate_video_async`, form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      const jobId = start.data.job_id as string
      setAnnotateProgress(p => ({ ...p, jobId }))

      pollTimer.current = window.setInterval(async () => {
        try {
          const { data } = await axios.get(`${API_BASE}/annotate_progress/${jobId}`)
          setAnnotateProgress({ jobId, percent: data.percent ?? null, processed: data.processed ?? 0, total: data.total ?? null })
          if (data.status === 'done') {
            clearAnnotatePolling()
            const resp = await axios.get(`${API_BASE}/annotate_result/${jobId}`, { responseType: 'blob' })
            const blob = new Blob([resp.data], { type: 'video/mp4' })
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = (file.name.replace(/\.[^/.]+$/, '') || 'annotated') + '_annotated.mp4'
            document.body.appendChild(a)
            a.click()
            a.remove()
            URL.revokeObjectURL(url)
            setBanner({ type: 'success', message: 'Annotated video downloaded.' })
            setAnalyzing(false)
            setAnnotateProgress({ jobId: null, percent: null, processed: 0, total: null })
          } else if (data.status === 'error') {
            clearAnnotatePolling()
            setBanner({ type: 'error', message: `Annotate failed: ${data.error || 'Unknown error'}` })
            setAnalyzing(false)
            setAnnotateProgress({ jobId: null, percent: null, processed: 0, total: null })
          }
        } catch (pollErr: any) {
          clearAnnotatePolling()
          setBanner({ type: 'error', message: `Annotate polling error: ${pollErr?.message || 'Unknown error'}` })
          setAnalyzing(false)
          setAnnotateProgress({ jobId: null, percent: null, processed: 0, total: null })
        }
      }, 800)
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Unknown error'
      setBanner({ type: 'error', message: `Annotate start failed: ${msg}` })
      setAnalyzing(false)
      setAnnotateProgress({ jobId: null, percent: null, processed: 0, total: null })
    } finally {
      e.target.value = ''
    }
  }

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif', padding: 20 }}>
      <h2>Human Fall Detection</h2>
      <div style={{ color: '#666' }}>Backend device: {device}</div>

      {banner && (
        <div style={{ marginTop: 12 }}>
          <Banner type={banner.type} message={banner.message} />
        </div>
      )}

      <div style={{ position: 'relative', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16, pointerEvents: analyzing ? 'none' : 'auto', opacity: analyzing ? 0.7 : 1 }}>
        {analyzing && <Spinner />}
        <div>
          <div style={{ position: 'relative', width: '100%' }}>
            <video ref={videoRef} style={{ width: '100%', height: 'auto', display: 'block', background: '#000' }} playsInline muted />
            <canvas ref={overlayRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
            <button onClick={startWebcam} disabled={analyzing || webcamActive}>Start webcam</button>
            <button onClick={stopWebcam} disabled={!webcamActive}>Stop webcam</button>
            <button onClick={() => captureAndAnalyze(false)} disabled={analyzing || !webcamActive}>Capture + Analyze</button>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <input type="checkbox" checked={livePose} onChange={(e) => setLivePose(e.target.checked)} disabled={!webcamActive} />
              Live Pose
            </label>
          </div>
          {lastResult && (
            <div style={{ marginTop: 8 }}>
              <strong>Fall:</strong> {lastResult.is_fall ? 'YES' : 'NO'} (score {lastResult.fall_score.toFixed(2)})
            </div>
          )}
        </div>
        <div>
          <div style={{ marginBottom: 8 }}>Analyze uploaded video:</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <input type="checkbox" checked={saveToBucket} onChange={(e) => setSaveToBucket(e.target.checked)} disabled={analyzing} />
              Save to OCI bucket
            </label>
          </div>
          <input type="file" accept="video/*" onChange={handleVideoUpload} disabled={analyzing} />
          <div style={{ marginTop: 12 }}>Annotate uploaded video (download):</div>
          <input type="file" accept="video/*" onChange={handleAnnotateVideo} disabled={analyzing} />
          {annotateProgress.jobId && (
            <div style={{ marginTop: 12 }}>
              <div style={{ marginBottom: 6 }}>Processing video… {annotateProgress.percent != null ? `${annotateProgress.percent.toFixed(0)}%` : `${annotateProgress.processed}${annotateProgress.total ? `/${annotateProgress.total}` : ''}`}</div>
              <div style={{ height: 8, background: '#eee', borderRadius: 6, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${annotateProgress.percent ?? 0}%`, background: '#3b82f6' }} />
              </div>
            </div>
          )}
          {lastVideoSummary && (
            <div style={{ marginTop: 8 }}>
              <strong>Video:</strong> {lastVideoSummary}
            </div>
          )}
        </div>
      </div>

      {lastRawJson && (
        <div style={{ marginTop: 16 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Last response</div>
          <pre style={{ background: '#0b1021', color: '#cbe1ff', padding: 12, borderRadius: 8, overflow: 'auto', maxHeight: 320 }}>{lastRawJson}</pre>
        </div>
      )}
    </div>
  )
}
