import { useCallback, useEffect, useRef, useState } from "react";
import type { Landmark, ServerMsg } from "../types";

export type WSStatus = "connecting" | "connected" | "disconnected";

export function useTranslatorWS() {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<WSStatus>("connecting");
  const [currentSign, setCurrentSign] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [glossas, setGlossas] = useState<string[]>([]);
  const [translation, setTranslation] = useState<string | null>(null);
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [isTranslating, setIsTranslating] = useState(false);

  const connect = useCallback(() => {
    const ws = new WebSocket(`ws://${location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => setStatus("connected");
    ws.onerror = () => ws.close();
    ws.onclose = () => {
      setStatus("disconnected");
      setTimeout(connect, 2000);
    };

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as ServerMsg;
      if (msg.type === "frame_result") {
        setLandmarks(msg.landmarks);
        setCurrentSign(msg.current_sign);
        setConfidence(msg.confidence);
        setGlossas(msg.glossas);
      } else if (msg.type === "translation") {
        setTranslation(msg.text);
        setIsTranslating(false);
      } else if (msg.type === "clear") {
        setGlossas([]);
        setTranslation(null);
        setIsTranslating(false);
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  const sendFrame = useCallback((b64: string) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "frame", frame: b64 }));
    }
  }, []);

  const translate = useCallback(() => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN && !isTranslating) {
      setIsTranslating(true);
      ws.send(JSON.stringify({ action: "translate" }));
    }
  }, [isTranslating]);

  const clear = useCallback(() => {
    wsRef.current?.readyState === WebSocket.OPEN &&
      wsRef.current.send(JSON.stringify({ action: "clear" }));
  }, []);

  return {
    status,
    currentSign,
    confidence,
    glossas,
    translation,
    landmarks,
    isTranslating,
    sendFrame,
    translate,
    clear,
  };
}
