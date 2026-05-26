import { CameraView } from "./components/CameraView";
import { ConnectionDot } from "./components/ConnectionDot";
import { SidePanel } from "./components/SidePanel";
import { useTranslatorWS } from "./hooks/useTranslatorWS";

export default function App() {
  const {
    status,
    currentSign,
    confidence,
    landmarks,
    glossas,
    translation,
    isTranslating,
    sendFrame,
    translate,
    clear,
  } = useTranslatorWS();

  return (
    <div className="bg-[#0f0f13] text-gray-200 h-dvh flex flex-col font-sans">
      <header className="px-6 py-3.5 bg-[#1a1a24] border-b border-[#2a2a3a] flex items-center gap-3 flex-shrink-0">
        <ConnectionDot status={status} />
        <h1 className="text-sm font-semibold tracking-wide">Tradutor LIBRAS</h1>
      </header>

      <main className="flex flex-1 overflow-hidden">
        <CameraView
          sendFrame={sendFrame}
          landmarks={landmarks}
          currentSign={currentSign}
          confidence={confidence}
        />
        <SidePanel
          glossas={glossas}
          translation={translation}
          isTranslating={isTranslating}
          onTranslate={translate}
          onClear={clear}
        />
      </main>
    </div>
  );
}
