// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

#include "mediapipe/framework/formats/landmark.pb.h"

#include<dirent.h>
#include<string>

using namespace std;


constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (save_video) {
    LOG(INFO) << "Prepare video writer.";
    cv::Mat test_frame;
    capture.read(test_frame);                    // Consume first frame.
    capture.set(cv::CAP_PROP_POS_AVI_RATIO, 0);  // Rewind to beginning.
    writer.open(FLAGS_output_video_path,
                mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                capture.get(cv::CAP_PROP_FPS), test_frame.size());
    RET_CHECK(writer.isOpened());
  } else {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller multi_hand_landmarks_poller,
				   graph.AddOutputStreamPoller("multi_hand_landmarks"));
  MP_RETURN_IF_ERROR(graph.StartRun({}));




  if (load_video) {
	
	LOG(INFO)<<FLAGS_input_video_path;
	string output=FLAGS_input_video_path;
	output.replace(output.end()-4,output.end(),".txt");    
	freopen( output.c_str(), "w", stdout );
	cout<<"LM0_00\tLM0_01\tLM0_02\tLM0_03\tLM0_04\tLM0_05\tLM0_06\tLM0_07\tLM0_08\tLM0_09\tLM0_10\tLM0_11\tLM0_12\tLM0_13\tLM0_14\tLM0_15\tLM0_16\tLM0_17\tLM0_18\tLM0_19\tLM0_20\t";
	cout<<"LM1_00\tLM1_01\tLM1_02\tLM1_03\tLM1_04\tLM1_05\tLM1_06\tLM1_07\tLM1_08\tLM1_09\tLM1_10\tLM1_11\tLM1_12\tLM1_13\tLM1_14\tLM1_15\tLM1_16\tLM1_17\tLM1_18\tLM1_19\tLM1_20\n";

  }
  LOG(INFO) << "Start grabbing and processing frames.";
  size_t frame_timestamp = 0;
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) break;  // End of video.
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp, &graph,
                                   &gpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp++))));
          return ::mediapipe::OkStatus();
        }));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;
	
	mediapipe::Packet multi_hand_landmarks_packet;
	if (!multi_hand_landmarks_poller.Next(&multi_hand_landmarks_packet)) break;
	//const auto& multi_hand_landmarks = multi_hand_landmarks_packet.Get<std::vector<std::vector<mediapipe::NormalizedLandmark>>>();
	const auto& multi_hand_landmarks = multi_hand_landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
	    
    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
          auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
          auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
          output_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
              gpu_frame.width(), gpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          gpu_helper.BindFramebuffer(texture);
          const auto info =
              mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
          glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                       info.gl_type, output_frame->MutablePixelData());
          glFlush();
          texture.Release();
          return ::mediapipe::OkStatus();
        }));

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
	
	int hand_index = 0;
	for (const auto& hand_landmarks : multi_hand_landmarks) {
	  int landmark_index = 0;
	  
	  for (const auto& landmark : hand_landmarks.landmark()) {

	   // std::cout << "[Hand<" << hand_index << ">] Landmark<" << landmark_index++ << ">: (" << landmark.x() << ", " << landmark.y() << ", " << landmark.z() << ")\n";
	  
	    LOG(INFO) << "[Hand<" << hand_index << ">]    Landmark<" << landmark_index++ << ">: (" << landmark.x() << ", " << landmark.y() << ", " << landmark.z() << ")\n";
	  	if(hand_index==0 && landmark_index==1)
	  		cout<<landmark.x()<<","<<landmark.y();	
	  	else
	  		cout<<"\t"<<landmark.x()<<","<<landmark.y();
	  }
		LOG(INFO) << hand_index;
	 	//std::cout << "\n";
	 	++hand_index;
	}
	if (hand_index==0)
		cout<<"0,0\n";
	else
		cout<<"\n";
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);
  
  if (argc==2)
  {
	  gflags::ParseCommandLineFlags(&argc, &argv, true);
	  ::mediapipe::Status run_status = RunMPPGraph();
	  if (!run_status.ok()) {
	    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
	  } else {
	    LOG(INFO) << "Success!";
	  }
	  return 0;	
  }
  else
  {

		DIR *pDIR;
        struct dirent *entry;
        if( pDIR=opendir("./videos/") )
        {
                while(entry = readdir(pDIR))
                {
                    if( string(entry->d_name).find(".mp4") <50)
                    {

                    	string folder="videos/";
                        LOG(INFO) << entry->d_name;	
                        string argument="--input_video_path="+folder+string(entry->d_name);
                        string name=folder+string(entry->d_name);
                        strcpy(argv[2],argument.c_str());
					    FLAGS_input_video_path=name;

					    gflags::ParseCommandLineFlags(&argc, &argv, true);
						::mediapipe::Status run_status = RunMPPGraph();
						if (!run_status.ok()) {
						  LOG(ERROR) << "Failed to run the graph: " << run_status.message();
						} else {
						  LOG(INFO) << "Success!";
						}
                    }
                }
                closedir(pDIR);
        }
	  
	  return 0;
  }
}
