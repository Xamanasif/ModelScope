import { createContext, useContext } from "react";
import { Json } from "./api";

export type Ctx = { status: Json; studyId: string | null; study: Json; setStudyId: (id: string) => void; refresh: () => void };
export const StudyContext = createContext<Ctx>({ status: null, studyId: null, study: null, setStudyId: () => {}, refresh: () => {} });
export const useStudy = () => useContext(StudyContext);
