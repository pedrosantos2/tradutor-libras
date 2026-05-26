export type Landmark = { x: number; y: number };

export type FrameResult = {
  type: "frame_result";
  landmarks: Landmark[];
  current_sign: string;
  confidence: number;
  glossas: string[];
};

export type Translation = { type: "translation"; text: string };
export type ClearMsg = { type: "clear" };
export type ServerMsg = FrameResult | Translation | ClearMsg;
