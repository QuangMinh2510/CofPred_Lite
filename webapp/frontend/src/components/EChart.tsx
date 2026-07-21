import * as echarts from "echarts";
import { useEffect, useRef } from "react";

interface Props {
  option: echarts.EChartsCoreOption;
  height?: number;
}

export default function EChart({ option, height = 420 }: Props) {
  const element = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current);
    chart.setOption(option, true);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [option]);

  return <div ref={element} style={{ width: "100%", height }} />;
}

