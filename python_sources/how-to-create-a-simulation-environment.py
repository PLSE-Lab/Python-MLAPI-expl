#!/usr/bin/env python
# coding: utf-8

# This example demonstrates how to create the [TicTacToe Environment](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/tictactoe) directly in a notebook.

# # 1. Import kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found. 
get_ipython().system('curl -X PURGE https://pypi.org/simple/kaggle-environments')

# 3. Register is in all versions of kaggle-environments
get_ipython().system("pip install 'kaggle-environments>=0.1.6'")

# 4. Import register to define the environment and make to create it.
from kaggle_environments import make, register


# # 2. Define Specification

# In[ ]:


specification = {
  "name": "tictactoe2",
  "title": "Tic Tac Toe",
  "description": "Classic Tic Tac Toe",
  "version": "1.0.0",
  "agents": [2],
  "configuration": {
    "steps": {
      "description": "Maximum number of steps the environment can run.",
      "type": "integer",
      "minimum": 10,
      "default": 10
    }
  },
  "reward": {
    "description": "0 = Lost, 0.5 = Draw, 1 = Won",
    "enum": [0, 0.5, 1],
    "default": 0.5
  },
  "observation": {
    "board": {
      "description": "Serialized 3x3 grid. 0 = Empty, 1 = X, 2 = O",
      "type": "array",
      "default": [0, 0, 0, 0, 0, 0, 0, 0, 0],
      "minItems": 9,
      "maxItems": 9
    },
    "mark": {
      "description": "Mark for the agent to use",
      "enum": [1, 2]
    }
  },
  "action": {
    "description": "Position to place a mark on the board.",
    "type": "integer",
    "minimum": 0,
    "maximum": 8
  },
  "reset": {
    "status": ["ACTIVE", "INACTIVE"],
    "observation": [{ "mark": 1 }, { "mark": 2 }],
    "reward": 0.5
  }
}


# # 3. Create State/Action Interpreter

# In[ ]:


# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from os import path
from random import choice

checks = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
]

EMPTY = 0

def interpreter(state, env):
    # Specification can fully handle the reset.
    if env.done:
        return state

    # Isolate the active and inactive agents.
    active = state[0] if state[0].status == "ACTIVE" else state[1]
    inactive = state[0] if state[0].status == "INACTIVE" else state[1]
    if active.status != "ACTIVE" or inactive.status != "INACTIVE":
        active.status = "DONE" if active.status == "ACTIVE" else active.status
        inactive.status = "DONE" if inactive.status == "INACTIVE" else inactive.status
        return state

    # Keep the board in sync between both agents.
    board = active.observation.board
    inactive.observation.board = board

    # Illegal move by the active agent.
    if board[active.action] != EMPTY:
        active.status = f"Invalid move: {active.action}"
        inactive.status = "DONE"
        return state

    # Mark the position.
    board[active.action] = active.observation.mark

    # Check for a win.
    if any(all(board[p] == active.observation.mark for p in c) for c in checks):
        active.reward = 1
        active.status = "DONE"
        inactive.reward = 0
        inactive.status = "DONE"
        return state

    # Check for a tie.
    if all(mark != EMPTY for mark in board):
        active.status = "DONE"
        inactive.status = "DONE"
        return state

    # Swap active and inactive agents to switch turns.
    active.status = "INACTIVE"
    inactive.status = "ACTIVE"

    return state


# # 4. Create Ansi Renderer 

# In[ ]:


def renderer(state, env):
    row_bar = "\n---+---+---\n"
    marks = [" ", "X", "O"]

    def print_pos(pos):
        str = ""
        if pos % 3 == 0 and pos > 0:
            str += row_bar
        if pos % 3 != 0:
            str += "|"
        return str + f" {marks[state[0].observation.board[pos]]} "

    return "".join(print_pos(p) for p in range(9))


# # 5. Create HTML Renderer (Optional)

# In[ ]:


get_ipython().run_cell_magic('writefile', 'html_renderer.js', '\nasync function renderer(context) {\n  const {\n    act,\n    agents,\n    environment,\n    frame,\n    height = 400,\n    interactive,\n    isInteractive,\n    parent,\n    step,\n    update,\n    width = 400,\n  } = context;\n\n  // Common Dimensions.\n  const canvasSize = Math.min(height, width);\n  const unit = 8;\n  const offset = canvasSize > 400 ? canvasSize * 0.1 : unit / 2;\n  const cellSize = (canvasSize - offset * 2) / 3;\n\n  // Canvas Setup.\n  let canvas = parent.querySelector("canvas");\n  if (!canvas) {\n    canvas = document.createElement("canvas");\n    parent.appendChild(canvas);\n\n    if (interactive) {\n      canvas.addEventListener("click", evt => {\n        if (!isInteractive()) return;\n        const rect = evt.target.getBoundingClientRect();\n        const x = evt.clientX - rect.left - offset;\n        const y = evt.clientY - rect.top - offset;\n        act(Math.floor(x / cellSize) + Math.floor(y / cellSize) * 3);\n      });\n    }\n  }\n  canvas.style.cursor = isInteractive() ? "pointer" : "default";\n\n  // Canvas setup and reset.\n  let c = canvas.getContext("2d");\n  canvas.width = canvasSize;\n  canvas.height = canvasSize;\n  c.clearRect(0, 0, canvas.width, canvas.height);\n\n  const drawStyle = ({\n    lineWidth = 1,\n    lineCap,\n    strokeStyle = "#FFF",\n    shadow,\n  }) => {\n    c.lineWidth = lineWidth;\n    c.strokeStyle = strokeStyle;\n    if (lineCap) c.lineCap = lineCap;\n    if (shadow) {\n      c.shadowOffsetX = shadow.offsetX || 0;\n      c.shadowOffsetY = shadow.offsetY || 0;\n      c.shadowColor = shadow.color || strokeStyle;\n      c.shadowBlur = shadow.blur || 0;\n    }\n  };\n\n  const drawLine = ({ x1, y1, x2, y2, style }) => {\n    c.beginPath();\n    drawStyle(style || {});\n    c.moveTo((x1 || 0) + offset, (y1 || 0) + offset);\n    c.lineTo((x2 || x1) + offset, (y2 || y1) + offset);\n    c.stroke();\n  };\n\n  const drawArc = ({ x, y, radius, sAngle, eAngle, style }) => {\n    drawStyle(style || {});\n    c.beginPath();\n    c.arc(x, y, radius, sAngle, eAngle);\n    c.stroke();\n  };\n\n  // Draw the Grid.\n  const gridFrame = step === 0 ? frame : 1;\n  const drawGridLine = ({\n    x1s = 0,\n    y1s = 0,\n    x2s,\n    y2s,\n    x1o = 0,\n    x2o = 0,\n    y1o = 0,\n    y2o = 0,\n  }) =>\n    drawLine({\n      x1: x1s * cellSize + x1o * unit,\n      x2: (x2s || x1s) * cellSize + x2o * unit,\n      y1: y1s * cellSize + y1o * unit,\n      y2: (y2s || y1s) * cellSize + y2o * unit,\n      style: { strokeStyle: "#046BBF" },\n    });\n\n  // Vertical.\n  drawGridLine({ x1s: 1, y1s: 0, y2s: gridFrame, y2o: -1 });\n  drawGridLine({ x1s: 2, y1s: 0, y2s: gridFrame, y2o: -1 });\n  drawGridLine({ x1s: 1, y1s: 1, y2s: 1 + gridFrame, y1o: 1, y2o: -1 });\n  drawGridLine({ x1s: 2, y1s: 1, y2s: 1 + gridFrame, y1o: 1, y2o: -1 });\n  drawGridLine({ x1s: 1, y1s: 2, y2s: 2 + gridFrame, y1o: 1 });\n  drawGridLine({ x1s: 2, y1s: 2, y2s: 2 + gridFrame, y1o: 1 });\n\n  // Horizontal.\n  drawGridLine({ x1s: 0, y1s: 1, x2s: gridFrame, x2o: -1 });\n  drawGridLine({ x1s: 1, y1s: 1, x2s: 1 + gridFrame, x1o: 1, x2o: -1 });\n  drawGridLine({ x1s: 2, y1s: 1, x2s: 2 + gridFrame, x1o: 1 });\n  drawGridLine({ x1s: 0, y1s: 2, x2s: gridFrame, x2o: -1 });\n  drawGridLine({ x1s: 1, y1s: 2, x2s: 1 + gridFrame, x1o: 1, x2o: -1 });\n  drawGridLine({ x1s: 2, y1s: 2, x2s: 2 + gridFrame, x1o: 1 });\n\n  // Draw the Pieces.\n  const drawX = (cell, cellFrame) => {\n    const part = cellSize / 4;\n    const gap = Math.min(Math.sqrt((unit * unit) / 2), canvasSize / 50);\n    const row = Math.floor(cell / 3);\n    const col = cell % 3;\n\n    const drawXLine = ({ x1, y1, x2, y2 }) =>\n      drawLine({\n        x1: col * cellSize + x1,\n        y1: row * cellSize + y1,\n        x2: col * cellSize + x2,\n        y2: row * cellSize + y2,\n        style: {\n          strokeStyle: "#FF0",\n          lineWidth: 2,\n          lineCap: "round",\n          shadow: { blur: 8 },\n        },\n      });\n\n    drawXLine({\n      x1: part,\n      y1: part,\n      x2: part + part * 2 * cellFrame,\n      y2: part + part * 2 * cellFrame,\n    });\n    if (Math.round(cellFrame) === 1) {\n      drawXLine({\n        x1: part,\n        y1: part * 3,\n        x2: part * 2 - gap,\n        y2: part * 2 + gap,\n      });\n      drawXLine({\n        x1: part * 2 + gap,\n        y1: part * 2 - gap,\n        x2: part * 3,\n        y2: part,\n      });\n    }\n  };\n  const drawO = (cell, cellFrame) => {\n    const row = Math.floor(cell / 3);\n    const col = cell % 3;\n    const radius = cellSize / 4 + 1; // +1 is for optical illusion.\n    const gap =\n      (Math.acos((2 * (radius ^ 2) - (unit ^ 2)) / (2 * radius * radius)) /\n        180) *\n      Math.PI *\n      unit;\n    const x = cellSize * col + cellSize / 2 + offset;\n    const y = cellSize * row + cellSize / 2 + offset;\n\n    const drawOArc = (sAngle, eAngle) =>\n      drawArc({\n        x,\n        y,\n        radius,\n        sAngle,\n        eAngle,\n        style: {\n          lineWidth: 2,\n          strokeStyle: "#F0F",\n          shadow: { blur: 8 },\n        },\n      });\n\n    drawOArc(\n      -Math.PI / 2 + gap,\n      -Math.PI / 2 + gap + (Math.PI - gap * 2) * cellFrame\n    );\n    drawOArc(\n      Math.PI / 2 + gap,\n      Math.PI / 2 + gap + (Math.PI - gap * 2) * cellFrame\n    );\n  };\n\n  const board = environment.steps[step][0].observation.board;\n\n  board.forEach((value, cell) => {\n    const cellFrame =\n      step <= 1 ||\n      environment.steps[step - 1][0].observation.board[cell] !== value\n        ? frame\n        : 1;\n    if (value === 1) drawX(cell, cellFrame);\n    if (value === 2) drawO(cell, cellFrame);\n  });\n\n  // Draw the winning line.\n  // [cell1, cell2, cell3, x1, y1, x2, y2]\n  const checks = [\n    [0, 1, 2, 1 / 19, 1 / 6, 18 / 19, 1 / 6],\n    [3, 4, 5, 1 / 19, 1 / 2, 18 / 19, 1 / 2],\n    [6, 7, 8, 1 / 19, 5 / 6, 18 / 19, 5 / 6],\n    [0, 3, 6, 1 / 6, 1 / 19, 1 / 6, 18 / 19],\n    [1, 4, 7, 1 / 2, 1 / 19, 1 / 2, 18 / 19],\n    [2, 5, 8, 5 / 6, 1 / 19, 5 / 6, 18 / 19],\n    [0, 4, 8, 1 / 19, 1 / 19, 18 / 19, 18 / 19],\n    [2, 4, 6, 18 / 19, 1 / 19, 1 / 19, 18 / 19],\n  ];\n  for (const check of checks) {\n    if (\n      board[check[0]] !== 0 &&\n      board[check[0]] === board[check[1]] &&\n      board[check[0]] === board[check[2]]\n    ) {\n      const x1 = check[3] * (cellSize * 3);\n      const y1 = check[4] * (cellSize * 3);\n      const winFrame = frame < 0.5 ? 0 : (frame - 0.5) / 0.5;\n      if (winFrame > 0) {\n        drawLine({\n          x1,\n          y1,\n          x2: x1 + (check[5] * (cellSize * 3) - x1) * winFrame,\n          y2: y1 + (check[6] * (cellSize * 3) - y1) * winFrame,\n          style: {\n            strokeStyle: "#FFF",\n            lineWidth: 3 * winFrame,\n            shadow: { blur: 8 * winFrame },\n          },\n        });\n      }\n      break;\n    }\n  }\n\n  // Upgrade the legend.\n  if (agents.length && (!agents[0].color || !agents[0].image)) {\n    const getPieceImage = drawFn => {\n      const pieceCanvas = document.createElement("canvas");\n      parent.appendChild(pieceCanvas);\n      pieceCanvas.style.marginLeft = "10000px";\n      pieceCanvas.width = cellSize + offset * 2;\n      pieceCanvas.height = cellSize + offset * 2;\n      c = pieceCanvas.getContext("2d");\n      drawFn(0, 1);\n      const dataUrl = pieceCanvas.toDataURL();\n      parent.removeChild(pieceCanvas);\n      return dataUrl;\n    };\n\n    agents.forEach(agent => {\n      agent.color = agent.index === 0 ? "#0FF" : "#FFF";\n      agent.image = getPieceImage(agent.index === 0 ? drawX : drawO);\n    });\n    update({ agents });\n  }\n}')


# In[ ]:


def html_renderer():
    with open("/kaggle/working/html_renderer.js") as f:
        return f.read()


# # 6. Include Default Agents (Optional)

# In[ ]:


from random import choice

checks = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
]

EMPTY = 0


def random_agent(obs):
    return choice([c for c in range(len(obs.board)) if obs.board[c] == EMPTY])


def reaction_agent(obs):
    # Connect 3 in a row to win.
    for check in checks:
        left = list(filter(lambda c: obs.board[c] != obs.mark, check))
        if len(left) == 1 and obs.board[left[0]] == EMPTY:
            return left[0]

    # Block 3 in a row to prevent loss.
    opponent = 2 if obs.mark == 1 else 1
    for check in checks:
        left = list(filter(lambda c: obs.board[c] != opponent, check))
        if len(left) == 1 and obs.board[left[0]] == EMPTY:
            return left[0]

    # No 3-in-a-rows, return random unmarked.
    return choice(list(filter(lambda m: m[1] == EMPTY, enumerate(obs.board))))[0]


agents = {"random": random_agent, "reaction": reaction_agent}


# # 7. Register as specification name

# In[ ]:


register(specification["name"], {
    "agents": agents,
    "html_renderer": html_renderer,
    "interpreter": interpreter,
    "renderer": renderer,
    "specification": specification,
})


# # 8. Make and Test the Environment

# In[ ]:


env = make(specification["name"], debug=True)
env.run(["random", "reaction"])
env.render(mode="ipython")

