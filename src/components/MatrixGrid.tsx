import type { Matrix } from '../cnn';

type MatrixGridProps = {
  matrix: Matrix;
  title?: string;
  caption?: string;
  showValues?: boolean;
  variant?: 'small' | 'medium' | 'large';
};

function clamp01(value: number): number {
  if (Number.isNaN(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function formatValue(value: number): string {
  if (Math.abs(value) >= 1 || value === 0) {
    return value.toFixed(1);
  }
  return value.toPrecision(2);
}

export default function MatrixGrid({
  matrix,
  title,
  caption,
  showValues = false,
  variant = 'medium',
}: MatrixGridProps) {
  if (!matrix.length || !matrix[0]?.length) {
    return null;
  }

  const columns = matrix[0].length;
  return (
    <div className={`matrix-grid matrix-grid--${variant}`}>
      {title ? <div className="matrix-grid__title">{title}</div> : null}
      <div className={`matrix-grid__cells matrix-grid__cells--${variant}`} style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {matrix.map((row, rowIndex) =>
          row.map((value, colIndex) => {
            const intensity = clamp01(value);
            const key = `${rowIndex}-${colIndex}`;
            return (
              <div
                key={key}
                className="matrix-grid__cell"
                style={{ backgroundColor: `rgba(17, 24, 39, ${intensity})` }}
              >
                {showValues ? (
                  <span
                    className="matrix-grid__value"
                    style={{ color: intensity > 0.5 ? '#ffffff' : '#111827' }}
                  >
                    {formatValue(value)}
                  </span>
                ) : null}
              </div>
            );
          }),
        )}
      </div>
      {caption ? <div className="matrix-grid__caption">{caption}</div> : null}
    </div>
  );
}

