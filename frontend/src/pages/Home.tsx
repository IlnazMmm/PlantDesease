import React, { useState } from "react";
import axios from "axios";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const upload = async () => {
    if (!file) return alert("Select a file");
    const fd = new FormData();
    fd.append("file", file);
    const res = await axios.post("http://localhost:8000/api/v1/upload", fd, { headers: { "Content-Type": "multipart/form-data" }});
    return res.data.file_id;
  };

  const handleRun = async () => {
    const file_id = await upload();
    const job = await axios.post("http://localhost:8000/api/v1/predict", { file_id, model_version: "v1" });
    setJobId(job.data.job_id);
    setStatus("queued");
    poll(job.data.job_id);
  };

  const poll = async (jid: string) => {
    const id = setInterval(async () => {
      const s = await axios.get(`http://localhost:8000/api/v1/status/${jid}`);
      setStatus(s.data.status);
      if (s.data.status === "done") {
        clearInterval(id);
        const r = await axios.get(`http://localhost:8000/api/v1/result/${jid}`);
        setResult(r.data);
      }
      if (s.data.status === "error") {
        clearInterval(id);
        setStatus("error");
      }
    }, 1000);
  };

  return (
    <div className="bg-white p-6 rounded shadow">
      <div className="mb-4">
        <input type="file" accept="image/*" onChange={(e)=> setFile(e.target.files?.[0] || null)} />
      </div>
      <div className="mb-4">
        <button className="bg-blue-600 text-white px-4 py-2 rounded" onClick={handleRun}>Analyze</button>
      </div>

      {jobId && <div className="mb-4">Job: {jobId} — Status: {status}</div>}

      {result && (
        <div className="mt-4">
          <h2 className="text-xl font-semibold">Result</h2>
          <p><strong>Plant:</strong> {result.plant}</p>
          <p><strong>Disease:</strong> {result.disease}</p>
          <p><strong>Confidence:</strong> {(result.confidence*100).toFixed(1)}%</p>
          {result.gradcam_url && <img src={`http://localhost:8000${result.gradcam_url}`} alt="gradcam" className="mt-2 max-w-full rounded shadow" />}
          <h3 className="mt-2 font-semibold">Description</h3>
          <p>{result.description}</p>
          <h3 className="mt-2 font-semibold">Treatment</h3>
          <p>{result.treatment}</p>
        </div>
      )}
    </div>
  )
}
