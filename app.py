import os
import cv2
import numpy as np
import time
import gradio as gr
ENVIRONMENT = "pi"
try:
  from sensor import Sensor
except ImportError:
  Sensor = lambda : None
  ENVIRONMENT = "dev"

from agent import Agent


sensor = Sensor()
pano_agent = Agent(sensor = sensor)

if ENVIRONMENT == "dev":
  with gr.Blocks() as demo:
      with gr.Row():
        # start_agent_btn = gr.Button("Submit")
        # stop_button = gr.Button("Stop Process", variant="stop")
        text_info = gr.Textbox(label="Information", value="Welcome to the Bird Detector Demo!")
      with gr.Row():
        input_img = gr.Image(label="webcam input", sources=["webcam"], type="numpy")
        dummy_img = gr.Image(visible=False)
        img_state = gr.State([])
        agent_view = gr.Image(label="Detection output", type="numpy")
        input_img.stream(
          lambda img, state: ((state[-1:] + [img]), 
                              pano_agent.work_once(state[-1:] + [img])),
          inputs=[input_img, img_state],
          outputs=[img_state, agent_view],
        )

if ENVIRONMENT == "pi":
  with gr.Blocks() as demo:
    with gr.Row():
      start_agent_btn = gr.Button("Submit")
      stop_button = gr.Button("Stop Process", variant="stop")
    with gr.Row():
      agent_view = gr.Image()
      agent_work = start_agent_btn.click(pano_agent.work,
                            inputs = None,
                            outputs = agent_view
                            )
      stop_button.click(
        fn=None,  # No function to execute on stop, just cancel
        inputs=None,
        outputs=None,
        cancels=[agent_work]
      )

demo.launch(
    # debug = True,
    server_name="0.0.0.0",
)
