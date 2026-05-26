import { useEffect, useRef } from "react";
import { useCamera } from "../hooks/useCamera";
import type { Landmark } from "../types";

type Props = {
  sendFrame: (b64: string) => void;
  landmarks: Landmark[];
  currentSign: string;
  confidence: number;
};

const FPS = 15;
const QUALITY = 0.85;

export function CameraView({ sendFrame, landmarks, currentSign, confidence }: Props) {
  const videoRef = useCamera();
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(null);

  // Capture loop — sends JPEG frames at 15fps
  useEffect(() => {
    const id = setInterval(() => {
      const video = videoRef.current;
      const cap = captureRef.current;
      if (!video || !cap || video.readyState < 2) return;
      cap.width = video.videoWidth;
      cap.height = video.videoHeight;
      cap.getContext("2d")!.drawImage(video, 0, 0);
      sendFrame(cap.toDataURL("image/jpeg", QUALITY).split(",")[1]);
    }, 1000 / FPS);
    return () => clearInterval(id);
  }, [sendFrame]);

  // Landmark overlay — mirrors x because video is CSS-flipped
  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas) return;
    const w = video?.videoWidth || 640;
    const h = video?.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#22c55e";
    landmarks.forEach(({ x, y }) => {
      ctx.beginPath();
      ctx.arc((1 - x) * w, y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [landmarks]);

  const pct = Math.round(confidence * 100);

  return (
    <div className="relative bg-black overflow-hidden flex-1">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full h-full object-cover"
        style={{ transform: "scaleX(-1)" }}
      />
      <canvas
        ref={overlayRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />
      <canvas ref={captureRef} className="hidden" />

      {/* Sign badge */}
      <div
        className={`absolute top-4 left-1/2 -translate-x-1/2 px-6 py-2 rounded-xl
          bg-black/65 backdrop-blur border border-white/10 text-white font-bold
          text-2xl tracking-widest min-w-[120px] text-center
          transition-opacity duration-200
          ${currentSign ? "opacity-100" : "opacity-30"}`}
      >
        {currentSign || "—"}
      </div>

      {/* Confidence bar */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 w-48">
        <div className="bg-black/50 rounded h-2 overflow-hidden">
          <div
            className="h-full rounded transition-[width] duration-150"
            style={{
              width: `${pct}%`,
              background: "linear-gradient(90deg,#6366f1,#22c55e)",
            }}
          />
        </div>
        <p className="text-center text-[0.7rem] text-gray-400 mt-1">
          {currentSign ? `${currentSign} — ${pct}%` : "aguardando sinal..."}
        </p>
      </div>
    </div>
  );
}
