import type { PointerEvent as ReactPointerEvent } from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Matrix, createMatrix } from '../cnn';

const MATRIX_SIZE = 28;

type DrawingPadProps = {
  size?: number;
  brushSize?: number;
  onMatrixChange: (matrix: Matrix) => void;
};

function createEmptyMatrix(): Matrix {
  return createMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
}

function sampleCanvasToMatrix(canvas: HTMLCanvasElement): Matrix {
  const context = canvas.getContext('2d');
  if (!context) {
    return createEmptyMatrix();
  }
  const { width, height } = canvas;
  const imageData = context.getImageData(0, 0, width, height);
  const data = imageData.data;
  const blockWidth = width / MATRIX_SIZE;
  const blockHeight = height / MATRIX_SIZE;
  const matrix = createEmptyMatrix();

  for (let row = 0; row < MATRIX_SIZE; row += 1) {
    const startY = Math.floor(row * blockHeight);
    const endY = Math.floor((row + 1) * blockHeight);
    for (let col = 0; col < MATRIX_SIZE; col += 1) {
      const startX = Math.floor(col * blockWidth);
      const endX = Math.floor((col + 1) * blockWidth);
      let accumulator = 0;
      let counter = 0;

      for (let y = startY; y < endY; y += 1) {
        for (let x = startX; x < endX; x += 1) {
          const index = (y * width + x) * 4;
          const r = data[index];
          const g = data[index + 1];
          const b = data[index + 2];
          const brightness = (r + g + b) / 3;
          const value = 1 - brightness / 255;
          accumulator += value;
          counter += 1;
        }
      }
      matrix[row][col] = counter ? accumulator / counter : 0;
    }
  }

  return matrix;
}

export default function DrawingPad({
  size = 280,
  brushSize = 18,
  onMatrixChange,
}: DrawingPadProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const requestRef = useRef<number | null>(null);
  const [matrix, setMatrix] = useState<Matrix>(() => createEmptyMatrix());
  const drawState = useRef({
    drawing: false,
    x: 0,
    y: 0,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = '#111111';
  }, [brushSize, size]);

  useEffect(() => {
    onMatrixChange(matrix);
  }, [matrix, onMatrixChange]);

  useEffect(
    () => () => {
      if (requestRef.current !== null) {
        cancelAnimationFrame(requestRef.current);
      }
    },
    [],
  );

  const scheduleMatrixUpdate = useCallback(() => {
    if (requestRef.current !== null) {
      return;
    }
    requestRef.current = requestAnimationFrame(() => {
      requestRef.current = null;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const sampled = sampleCanvasToMatrix(canvas);
      setMatrix(sampled);
    });
  }, []);

  const getCanvasPoint = useCallback((event: PointerEvent): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
    return { x, y };
  }, []);

  const drawLine = useCallback((from: { x: number; y: number }, to: { x: number; y: number }) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }, []);

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.setPointerCapture(event.pointerId);
      const point = getCanvasPoint(event.nativeEvent);
      if (!point) return;
      drawState.current = { drawing: true, x: point.x, y: point.y };
      drawLine(point, point);
      scheduleMatrixUpdate();
    },
    [drawLine, getCanvasPoint, scheduleMatrixUpdate],
  );

  const handlePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      if (!drawState.current.drawing) return;
      const point = getCanvasPoint(event.nativeEvent);
      if (!point) return;
      const { x, y } = drawState.current;
      drawLine({ x, y }, point);
      drawState.current = { drawing: true, x: point.x, y: point.y };
      scheduleMatrixUpdate();
    },
    [drawLine, getCanvasPoint, scheduleMatrixUpdate],
  );

  const endDrawing = useCallback(
    (event: ReactPointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (canvas && canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      if (!drawState.current.drawing) return;
      drawState.current = { drawing: false, x: 0, y: 0 };
      scheduleMatrixUpdate();
    },
    [scheduleMatrixUpdate],
  );

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = '#111111';
    setMatrix(createEmptyMatrix());
    scheduleMatrixUpdate();
  }, [brushSize, scheduleMatrixUpdate]);

  return (
    <div className="drawing-pad">
      <canvas
        ref={canvasRef}
        className="drawing-canvas"
        width={size}
        height={size}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrawing}
        onPointerLeave={endDrawing}
      />
      <div className="drawing-actions">
        <button type="button" onClick={clearCanvas}>
          Clear
        </button>
      </div>
    </div>
  );
}
