import "./App.css";
import {
  ScatterChart,
  Scatter,
  ResponsiveContainer,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
  Cell,
} from "recharts";
import { embeddings } from "./data";
import { useState } from "react";

const max = Math.max(...embeddings.map((e) => e.score));
const min = Math.min(...embeddings.map((e) => e.score));

function colorFromDataRange(x: number, range: [number, number]) {
  // generate color in rgb format using purple to yellow color scale
  const yellowRGB = [246, 252, 61];
  const purpleRGB = [62, 0, 168];

  const [min, max] = range;
  const normalized = (x - min) / (max - min);
  const oneMinus = 1 - normalized;

  const color = yellowRGB.map((c, i) => {
    return Math.round(c * normalized + purpleRGB[i] * oneMinus);
  });
  return `rgb(${color.join(",")})`;
}

function App() {
  const [index, setIndex] = useState(-1);

  return (
    <div
      style={{
        height: "100%",
        display: "flex",
        width: "100%",
        justifyItems: "space-between",
        alignItems: "center",
      }}
    >
      <div
        className=""
        style={{
          flex: 1,
        }}
      >
        <ResponsiveContainer width="100%" height={800}>
          <ScatterChart>
            <CartesianGrid />
            <XAxis type={"number"} dataKey={"x"} />
            <YAxis type={"number"} dataKey={"y"} />
            {/* z axis with heatmap as color of the dot*/}
            <ZAxis type={"number"} dataKey={"score"} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />
            <Scatter
              data={embeddings}
              onClick={(data, index) => {
                console.log(data, index);
                setIndex(index);
              }}
            >
              {embeddings.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={colorFromDataRange(entry.score, [min, max])}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div
        className=""
        style={{
          flex: 1,
        }}
      >
        {/* load image */}
        {index !== -1 && <img src={`/boards/${index}.svg`} alt="board" />}

        {/* load text */}
        {index !== -1 && (
          <p>
            <b>white score: </b>
            {embeddings[index].score}
          </p>
        )}
      </div>
    </div>
  );
}

export default App;
