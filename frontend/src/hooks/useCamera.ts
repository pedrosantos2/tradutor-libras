import { useEffect, useRef } from "react";

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;

    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then((s) => {
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch((err) => alert("Câmera não disponível: " + err.message));

    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  return videoRef;
}
