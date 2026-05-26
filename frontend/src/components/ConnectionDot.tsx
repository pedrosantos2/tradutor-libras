import type { WSStatus } from "../hooks/useTranslatorWS";

export function ConnectionDot({ status }: { status: WSStatus }) {
  return (
    <span
      className={`w-2.5 h-2.5 rounded-full flex-shrink-0 transition-colors ${
        status === "connected"
          ? "bg-green-500 shadow-[0_0_6px_#22c55e88]"
          : "bg-gray-500"
      }`}
    />
  );
}
