import { useEffect, useRef } from "react";

/**
 * setInterval that only fires while the tab is visible, plus once on becoming visible.
 * A hidden tab (phone in a pocket, background window) must not keep the free Render API
 * awake — every request resets its 15-minute sleep timer.
 */
export function useVisibleInterval(fn, ms, enabled = true) {
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!enabled) return undefined;
    const tick = () => {
      if (document.visibilityState === "visible") fnRef.current();
    };
    tick();
    const id = window.setInterval(tick, ms);
    document.addEventListener("visibilitychange", tick);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [ms, enabled]);
}
