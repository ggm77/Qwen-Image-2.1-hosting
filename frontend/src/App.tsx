// src/App.tsx
import React, { useState, useEffect, useRef } from "react";

type GenerateRequest = {
  prompt: string;
  height: number;
  width: number;
  num_inference_steps: number;
  seed: number;
};

type TaskStatus = {
  task_id: string;
  status: string;
  created_at: string;
  current_step: number;
  num_inference_steps: number;
};

const POLL_INTERVAL_MS = 1500;

const App: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [height, setHeight] = useState(768);
  const [width, setWidth] = useState(768);
  const [numSteps, setNumSteps] = useState(40);
  const [seed, setSeed] = useState(42);

  const [taskId, setTaskId] = useState("");
  const [lookupTaskId, setLookupTaskId] = useState("");
  const [taskStatus, setTaskStatus] = useState("");
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  // 언마운트 시 폴링 타이머 정리
  useEffect(() => {
    return () => stopPolling();
  }, []);

  const stopPolling = () => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const fetchImage = async (id: string) => {
    try {
      const res = await fetch(`/api/v1/tasks/${id}/result`, { method: "GET" });
      if (!res.ok) {
        throw new Error(`이미지 생성 결과 요청 실패: ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      setImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "알 수 없는 오류입니다.");
    }
  };

  // task_id 하나의 진행 상황을 주기적으로 조회하며 진행률을 갱신한다.
  const startPolling = (id: string) => {
    stopPolling();

    const poll = async () => {
      try {
        const res = await fetch(`/api/v1/tasks/${id}`, { method: "GET" });
        if (!res.ok) {
          throw new Error(`태스크 상태 요청 실패: ${res.status}`);
        }

        const data: TaskStatus = await res.json();
        setTaskStatus(data.status);
        setCurrentStep(data.current_step);
        setTotalSteps(data.num_inference_steps);

        if (data.status === "completed") {
          stopPolling();
          setIsGenerating(false);
          await fetchImage(id);
        } else if (data.status === "failed") {
          stopPolling();
          setIsGenerating(false);
          setError("태스크가 실패했습니다.");
        }
      } catch (err) {
        console.error(err);
        stopPolling();
        setIsGenerating(false);
        setError(err instanceof Error ? err.message : "알 수 없는 오류입니다.");
      }
    };

    poll(); // 즉시 1회 조회 후 주기 폴링 시작
    pollTimerRef.current = window.setInterval(poll, POLL_INTERVAL_MS);
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!prompt.trim()) {
      setError("prompt를 입력하세요.");
      return;
    }

    setError(null);
    setImageUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setCurrentStep(0);
    setTotalSteps(numSteps);

    try {
      const payload: GenerateRequest = {
        prompt,
        height,
        width,
        num_inference_steps: numSteps,
        seed,
      };

      const res = await fetch(`/api/v1/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`이미지 생성 요청 실패: ${res.status}`);
      }

      const data = await res.json();
      setTaskId(data.task_id);
      setTaskStatus(data.status);
      setIsGenerating(true);

      startPolling(data.task_id);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "알 수 없는 오류입니다.");
      setIsGenerating(false);
    }
  };

  // 입력한 task_id로 진행 상황을 조회하고, 완료 상태면 이어서 폴링한다.
  const handleLookup = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!lookupTaskId.trim()) {
      setError("조회할 task_id를 입력하세요.");
      return;
    }

    setError(null);
    setTaskId(lookupTaskId.trim());
    setIsGenerating(true);
    startPolling(lookupTaskId.trim());
  };

  const progressPercent =
    totalSteps > 0 ? Math.min(100, Math.round((currentStep / totalSteps) * 100)) : 0;

  return (
    <div style={{ padding: "20px", fontFamily: "sans-serif" }}>
      <h1>Simple Image Generator</h1>

      <form onSubmit={handleGenerate} style={{ display: "flex", flexDirection: "column", gap: "10px", maxWidth: 500 }}>

        <label>
          Prompt
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            style={{ width: "100%" }}
          />
        </label>

        <label>
          Height
          <input type="number" value={height} onChange={(e) => setHeight(Number(e.target.value))} />
        </label>

        <label>
          Width
          <input type="number" value={width} onChange={(e) => setWidth(Number(e.target.value))} />
        </label>

        <label>
          Num inference steps
          <input type="number" value={numSteps} onChange={(e) => setNumSteps(Number(e.target.value))} />
        </label>

        <label>
          Seed
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </label>

        <button type="submit" disabled={isGenerating}>
          {isGenerating ? "생성 중..." : "이미지 생성"}
        </button>
      </form>

      <form onSubmit={handleLookup} style={{ display: "flex", gap: "10px", marginTop: "20px", maxWidth: 500 }}>
        <input
          type="text"
          placeholder="조회할 task_id 입력"
          value={lookupTaskId}
          onChange={(e) => setLookupTaskId(e.target.value)}
          style={{ flex: 1 }}
        />
        <button type="submit">진행상황 조회</button>
      </form>

      <div style={{ marginTop: "20px" }}>
        <p>{taskId.length > 0 ? "태스크 ID: " + taskId : "태스크 ID: 태스크 조회 되지 않음"}</p>
        <p>{taskStatus.length > 0 ? "태스크 상태: " + taskStatus : "태스크 상태: 태스크 조회 되지 않음"}</p>

        {taskId.length > 0 && (
          <div style={{ maxWidth: 500 }}>
            <div
              style={{
                width: "100%",
                height: "20px",
                backgroundColor: "#eee",
                borderRadius: "4px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${progressPercent}%`,
                  height: "100%",
                  backgroundColor: taskStatus === "failed" ? "#e57373" : "#4caf50",
                  transition: "width 0.3s ease",
                }}
              />
            </div>
            <p>
              {totalSteps > 0
                ? `${currentStep} / ${totalSteps} 스텝 (${progressPercent}%)`
                : "대기 중..."}
            </p>
          </div>
        )}
      </div>

      {error && <p style={{ color: "red" }}>{error}</p>}

      <div style={{ marginTop: "20px" }}>
        {imageUrl ? (
          <img src={imageUrl} alt="generated" style={{ maxWidth: "100%" }} />
        ) : (
          <p>아직 생성된 이미지 없음.</p>
        )}
      </div>
    </div>
  );
};

export default App;
