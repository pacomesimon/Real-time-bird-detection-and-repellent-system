import gradio as gr
import json
import os
import subprocess
import signal
from sensor import Sensor, MultiSensor
from agent import Agent
import time

def load_full_config():
    with open("config.json", "r") as f:
        return json.load(f)

CONFIG_FULL = load_full_config()
CONFIG_GUI = CONFIG_FULL["gui_app"]

# Initialize sensors using the Sensor class from sensor.py
cam1 = Sensor(1)
cam2 = Sensor(0)

sensor = MultiSensor([cam1, cam2])
pano_agent = Agent(sensor=sensor)

def update_cam1(focus):
    cam1.adjust_focus(focus)
    return cam1.take_photo()

def update_cam2(focus):
    cam2.adjust_focus(focus)
    return cam2.take_photo()

def save_settings(default_focus, cached_images_dir, threshold, percentile, min_bbox_dim, save_images_bool_path, predictions_log_path):
    global cam1,cam2, pano_agent
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        # Update Sensor settings
        config["sensor"]["default_focus"] = float(default_focus)
        update_cam1(float(default_focus))
        update_cam2(float(default_focus))
        config["sensor"]["cached_images_dir"] = cached_images_dir
        
        # Update Bird Detection settings
        config["bird_detection"]["threshold"] = float(threshold)
        pano_agent.bird_detector.threshold = float(threshold)

        config["bird_detection"]["percentile"] = int(percentile)
        pano_agent.bird_detector.percentile = float(percentile)

        config["bird_detection"]["min_bbox_dim"] = int(min_bbox_dim)
        pano_agent.bird_detector.min_bbox_dim = float(min_bbox_dim)
        
        # Update CLI App settings
        config["cli_app"]["save_images_bool_path"] = save_images_bool_path
        config["cli_app"]["predictions_log_path"] = predictions_log_path
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        return "✅ Settings saved and config.json updated!"
    except Exception as e:
        return f"❌ Error saving settings: {str(e)}"

def read_save_images_bool():
    try:
        with open(CONFIG_FULL["cli_app"]["save_images_bool_path"], "r") as f:
            val = f.read().strip()
            return val == "1"
    except Exception:
        return False

def write_save_images_bool(val):
    try:
        with open(CONFIG_FULL["cli_app"]["save_images_bool_path"], "w") as f:
            f.write("1" if val else "0")
    except Exception as e:
        print("Error writing save_images_bool.txt: ", e)

def run_cli_process():
    global cam1, cam2, pano_agent
    try:
        # Close sensors before starting CLI app
        if cam1:
            try: cam1.picam2.close()
            except: pass
        if cam2:
            try: cam2.picam2.close()
            except: pass
        
        # Load current config
        with open("config.json", "r") as f:
            config = json.load(f)
            
        # Kill existing process if any
        old_pid = config.get("cli_app", {}).get("process_id")
        if old_pid:
            try:
                os.kill(int(old_pid), signal.SIGKILL)
                time.sleep(1)
            except:
                pass

        # Start new process in background
        cwd = os.path.dirname(os.path.abspath(__file__))
        process = subprocess.Popen(["python3", "cli_app.py"], cwd=cwd)
        new_pid = process.pid
        
        # Save new process ID to config
        config["cli_app"]["process_id"] = new_pid
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        return f"Running (PID: {new_pid})"
    except Exception as e:
        return f"Error: {str(e)}"

def stop_cli_process():
    global cam1, cam2, pano_agent
    try:
        # Load current config
        with open("config.json", "r") as f:
            config = json.load(f)
            
        # Kill process if ID is not null
        pid = config.get("cli_app", {}).get("process_id")
        if pid:
            try:
                os.system(f"kill -15 {int(pid)}")
                time.sleep(1)
                # Force kill if still alive
                try:
                    os.system(f"kill -9 {int(pid)}")
                    time.sleep(1)
                except OSError:
                    pass # Already gone
            except Exception as e:
                print(f"Error killing process {pid}: {e}")
                
            # Set process ID to null
            config["cli_app"]["process_id"] = None
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
            
            # Re-initialize sensors for GUI
            try:
                cam1 = Sensor(1)
                cam2 = Sensor(0)
                pano_agent = Agent(sensor=MultiSensor([cam1, cam2]))
            except Exception as e:
                return f"Process Stopped, but Sensor Error: {str(e)}"
                
            return "Process Stopped & Sensors Re-initialized"
        else:
            return "No process running"
    except Exception as e:
        return f"Error stopping process: {str(e)}"

def run_agent_work():
    global pano_agent
    return pano_agent.work()

def get_latest_image():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        img_dir = config["cli_app"]["cached_images_dir"]
        if not os.path.exists(img_dir):
            return None
        
        files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            return None
        
        # Get the latest file by modification time
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Error getting latest image: {e}")
        return None

def stream_latest_images():
    while True:
        img_path = get_latest_image()
        if img_path:
            yield img_path
        time.sleep(1) # Refresh every second


# Custom CSS for modern look
custom_css = """
.container { max-width: 1200px; margin: auto; padding-top: 20px; }
.save-btn { background-color: #2ecc71 !important; color: white !important; font-weight: bold !important; }
.header { text-align: center; margin-bottom: 2rem; }
.tab-content { border-radius: 10px; margin-top: 10px; }
"""

with gr.Blocks(
    css=custom_css, 
    theme=gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]
    )
    ) as demo:
    gr.Markdown("# 🦅 Dual Camera Bird Detection System", elem_classes="header")
    
    save_images_bool_checkbox = gr.Checkbox(
        label="Save Images (CLI App Option)",
        value=read_save_images_bool(),
        interactive=True
    )
    
    save_images_bool_checkbox.change(
        fn=lambda checked: write_save_images_bool(checked),
        inputs=save_images_bool_checkbox,
        outputs=None
    )
    with gr.Tabs():
        with gr.TabItem("🎥 Focus Adjustment"):
            with gr.Row():
                gr.Markdown("""
                ```
                NOTE: this is for preview only. Focus configuration can be changed in the next tab.
                ```
                """)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Camera 1")
                    slider1 = gr.Slider(
                        CONFIG_GUI["focus_slider"]["min"], 
                        CONFIG_GUI["focus_slider"]["max"], 
                        value=CONFIG_FULL["sensor"]["default_focus"], 
                        step=CONFIG_GUI["focus_slider"]["step"],
                        label="Focus Level"
                    )
                    image1 = gr.Image(type="numpy", label="Cam 1 Feed")
                with gr.Column():
                    gr.Markdown("### Camera 2")
                    slider2 = gr.Slider(
                        CONFIG_GUI["focus_slider"]["min"], 
                        CONFIG_GUI["focus_slider"]["max"], 
                        value=CONFIG_FULL["sensor"]["default_focus"], 
                        step=CONFIG_GUI["focus_slider"]["step"],
                        label="Focus Level"
                    )
                    image2 = gr.Image(type="numpy", label="Cam 2 Feed")
            
            slider1.change(update_cam1, slider1, image1)
            slider2.change(update_cam2, slider2, image2)

        with gr.TabItem("⚙️ Sensor Settings"):
            with gr.Column(elem_classes="tab-content"):
                s_focus = gr.Number(label="Default Focus", value=CONFIG_FULL["sensor"]["default_focus"])
                s_cache = gr.Textbox(label="Cached Images Directory", value=CONFIG_FULL["sensor"]["cached_images_dir"])

        with gr.TabItem("🔍 Bird Detection"):
            with gr.Column(elem_classes="tab-content"):
                bd_thresh = gr.Number(label="Detection Threshold", value=CONFIG_FULL["bird_detection"]["threshold"])
                bd_perc = gr.Number(label="Percentile Filter", value=CONFIG_FULL["bird_detection"]["percentile"])
                bd_min_dim = gr.Number(label="Minimum Bbox Dimension", value=CONFIG_FULL["bird_detection"]["min_bbox_dim"])

        with gr.TabItem("💻 CLI App"):
            with gr.Column(elem_classes="tab-content"):
                cli_save = gr.Textbox(label="Save Images Bool Path", value=CONFIG_FULL["cli_app"]["save_images_bool_path"])
                cli_log = gr.Textbox(label="Predictions Log Path", value=CONFIG_FULL["cli_app"]["predictions_log_path"])
        with gr.TabItem("⏳ Real time preview"):
            with gr.Row():
                start_agent_btn = gr.Button("Submit")
                stop_button = gr.Button("Stop Process", variant="stop")
            with gr.Row():
                agent_view = gr.Image()
                agent_work = start_agent_btn.click(run_agent_work,
                                    inputs = None,
                                    outputs = agent_view
                                    )
                stop_button.click(
                    fn=None,  # No function to execute on stop, just cancel
                    inputs=None,
                    outputs=None,
                    cancels=[agent_work]
                )
        with gr.TabItem("🚀 Process Control"):
            with gr.Column(elem_classes="tab-content"):
                with gr.Row():
                    run_cli_btn = gr.Button("🚀 Run CLI App", variant="primary")
                    stop_cli_btn = gr.Button("🛑 Stop CLI App", variant="stop")
                
                cli_status_area = gr.TextArea(
                    label="CLI Process Status", 
                    value="Ready" if CONFIG_FULL["cli_app"].get("process_id") is None else f"Running (PID: {CONFIG_FULL['cli_app']['process_id']})",
                    interactive=False
                )
                
                run_cli_btn.click(
                    fn=run_cli_process,
                    inputs=None,
                    outputs=cli_status_area
                )
                stop_cli_btn.click(
                    fn=stop_cli_process,
                    inputs=None,
                    outputs=cli_status_area
                )

        with gr.TabItem("🖼️ Latest Detections"):
            with gr.Column(elem_classes="tab-content"):
                gr.Markdown("### 📸 Live Feed from Cached Images")
                with gr.Row():
                    start_stream_btn = gr.Button("▶️ Start Stream", variant="primary")
                    stop_stream_btn = gr.Button("⏹️ Stop Stream", variant="stop")
                
                stream_display = gr.Image(label="Latest Detection")
                
                stream_event = start_stream_btn.click(
                    fn=stream_latest_images,
                    inputs=None,
                    outputs=stream_display
                )
                
                stop_stream_btn.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    cancels=[stream_event]
                )
        
    
        
    with gr.Row():
        save_btn = gr.Button("💾 Save Configuration", elem_classes="save-btn")
        save_msg = gr.Markdown("")

    save_btn.click(
        save_settings,
        inputs=[s_focus, s_cache, bd_thresh, bd_perc, bd_min_dim, cli_save, cli_log],
        outputs=save_msg
    )

# Run the demo
demo.launch(
    share=CONFIG_GUI["share"],
    server_name=CONFIG_GUI["server_name"],
)
