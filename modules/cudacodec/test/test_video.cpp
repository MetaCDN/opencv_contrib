/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"
namespace opencv_test {
    namespace {

#if defined(HAVE_NVCUVID) || defined(HAVE_NVCUVENC)
PARAM_TEST_CASE(Video, cv::cuda::DeviceInfo, std::string)
{
};

<<<<<<< HEAD
=======
typedef tuple<std::string, bool> color_conversion_params_t;
PARAM_TEST_CASE(ColorConversion, cv::cuda::DeviceInfo, cv::cudacodec::ColorFormat, color_conversion_params_t)
{
};

struct ReconfigureDecoderWithScaling : SetDevice
{
};

PARAM_TEST_CASE(ReconfigureDecoder, cv::cuda::DeviceInfo, int)
{
};

PARAM_TEST_CASE(VideoReadRaw, cv::cuda::DeviceInfo, std::string)
{
};

typedef tuple<std::string, bool> histogram_params_t;
PARAM_TEST_CASE(Histogram, cv::cuda::DeviceInfo, histogram_params_t)
{
};

PARAM_TEST_CASE(CheckKeyFrame, cv::cuda::DeviceInfo, std::string)
{
};

PARAM_TEST_CASE(CheckDecodeSurfaces, cv::cuda::DeviceInfo, std::string)
{
};

PARAM_TEST_CASE(CheckInitParams, cv::cuda::DeviceInfo, std::string, bool, bool, bool)
{
};

struct CheckParams : SetDevice
{
};

struct Seek : SetDevice
{
};

>>>>>>> 80f1ca2442982ed518076cd88cf08c71155b30f6
#if defined(HAVE_NVCUVID)
//////////////////////////////////////////////////////
// VideoReader

CUDA_TEST_P(Video, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());

    // CUDA demuxer has to fall back to ffmpeg to process "cv/video/768x576.avi"
    if (GET_PARAM(1) == "cv/video/768x576.avi" && !videoio_registry::hasBackend(CAP_FFMPEG))
        throw SkipTestException("FFmpeg backend not found");

    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
<<<<<<< HEAD
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    cv::cudacodec::FormatInfo fmt = reader->format();
    cv::cuda::GpuMat frame;
    for (int i = 0; i < 100; i++)
=======
    const string fileNameOut = tempfile("test_container_stream");
    {
        std::ofstream file(fileNameOut, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        cv::cudacodec::VideoReaderInitParams params;
        params.rawMode = true;
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
        double rawIdxBase = -1;
        ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_RAW_PACKAGES_BASE_INDEX, rawIdxBase));
        ASSERT_EQ(rawIdxBase, 2);
        cv::cuda::GpuMat frame;
        for (int i = 0; i < 100; i++)
        {
            ASSERT_TRUE(reader->grab());
            ASSERT_TRUE(reader->retrieve(frame));
            ASSERT_FALSE(frame.empty());
            double N = -1;
            ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB,N));
            ASSERT_TRUE(N >= 0) << N << " < 0";
            for (int j = static_cast<int>(rawIdxBase); j <= static_cast<int>(N + rawIdxBase); j++) {
                Mat rawPackets;
                reader->retrieve(rawPackets, j);
                file.write((char*)rawPackets.data, rawPackets.total());
            }
        }
    }

    std::cout << "Checking written video stream: " << fileNameOut << std::endl;

    {
        cv::Ptr<cv::cudacodec::VideoReader> readerReference = cv::cudacodec::createVideoReader(inputFile);
        cv::cudacodec::VideoReaderInitParams params;
        params.rawMode = true;
        cv::Ptr<cv::cudacodec::VideoReader> readerActual = cv::cudacodec::createVideoReader(fileNameOut, {}, params);
        double decodedFrameIdx = -1;
        ASSERT_TRUE(readerActual->get(cv::cudacodec::VideoReaderProps::PROP_DECODED_FRAME_IDX, decodedFrameIdx));
        ASSERT_EQ(decodedFrameIdx, 0);
        cv::cuda::GpuMat reference, actual;
        cv::Mat referenceHost, actualHost;
        for (int i = 0; i < 100; i++)
        {
            ASSERT_TRUE(readerReference->nextFrame(reference));
            ASSERT_TRUE(readerActual->grab());
            ASSERT_TRUE(readerActual->retrieve(actual, static_cast<size_t>(decodedFrameIdx)));
            actual.download(actualHost);
            reference.download(referenceHost);
            ASSERT_TRUE(cvtest::norm(actualHost, referenceHost, NORM_INF) == 0);
        }
    }

    ASSERT_EQ(0, remove(fileNameOut.c_str()));
}

CUDA_TEST_P(Histogram, Reader)
{
    cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + get<0>(GET_PARAM(1));
    const bool histAvailable = get<1>(GET_PARAM(1));
    cudacodec::VideoReaderInitParams params;
    params.enableHistogram = histAvailable;
    Ptr<cudacodec::VideoReader> reader;
    try {
        reader = cudacodec::createVideoReader(inputFile, {}, params);
    }
    catch (const cv::Exception& e) {
        throw SkipTestException(e.msg);
    }
    const cudacodec::FormatInfo fmt = reader->format();
    ASSERT_EQ(histAvailable, fmt.enableHistogram);
    reader->set(cudacodec::ColorFormat::GRAY);
    GpuMat frame, hist;
    reader->nextFrame(frame, hist);
    if (histAvailable) {
        ASSERT_TRUE(!hist.empty());
        Mat frameHost, histGsHostFloat, histGs, histHost;
        frame.download(frameHost);
        const int histSize = 256;
        const float range[] = { 0, 256 };
        const float* histRange[] = { range };
        cv::calcHist(&frameHost, 1, 0, Mat(), histGsHostFloat, 1, &histSize, histRange);
        histGsHostFloat.convertTo(histGs, CV_32S);
        if (fmt.videoFullRangeFlag)
            hist.download(histHost);
        else
            cudacodec::MapHist(hist, histHost);
        const double err = cv::norm(histGs.t(), histHost, NORM_INF);
        ASSERT_EQ(err, 0);
    }
    else {
        ASSERT_TRUE(hist.empty());
    }
}

CUDA_TEST_P(CheckParams, Reader)
{
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    {
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
        double msActual = -1;
        ASSERT_FALSE(reader->get(cv::VideoCaptureProperties::CAP_PROP_OPEN_TIMEOUT_MSEC, msActual));
    }

    {
        constexpr int msReference = 3333;
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {
            cv::VideoCaptureProperties::CAP_PROP_OPEN_TIMEOUT_MSEC, msReference });
        double msActual = -1;
        ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_OPEN_TIMEOUT_MSEC, msActual));
        ASSERT_EQ(msActual, msReference);
    }
}

CUDA_TEST_P(CheckParams, CaptureProps)
{
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
    double width, height, fps, iFrame;
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, width));
    ASSERT_EQ(672, width);
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, height));
    ASSERT_EQ(384, height);
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_FPS, fps));
    ASSERT_EQ(24, fps);
    ASSERT_TRUE(reader->grab());
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, iFrame));
    ASSERT_EQ(iFrame, 1.);
}

CUDA_TEST_P(CheckDecodeSurfaces, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
    int ulNumDecodeSurfaces = 0;
>>>>>>> 80f1ca2442982ed518076cd88cf08c71155b30f6
    {
        ASSERT_TRUE(reader->nextFrame(frame));
        if(!fmt.valid)
            fmt = reader->format();
        ASSERT_TRUE(frame.cols == fmt.width && frame.rows == fmt.height);
        ASSERT_FALSE(frame.empty());
    }
}
<<<<<<< HEAD
#endif // HAVE_NVCUVID

#if defined(_WIN32) && defined(HAVE_NVCUVENC)
=======

CUDA_TEST_P(CheckInitParams, Reader)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../" + GET_PARAM(1);
    cv::cudacodec::VideoReaderInitParams params;
    params.udpSource = GET_PARAM(2);
    params.allowFrameDrop = GET_PARAM(3);
    params.rawMode = GET_PARAM(4);
    double udpSource = 0, allowFrameDrop = 0, rawMode = 0;
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_UDP_SOURCE, udpSource) && static_cast<bool>(udpSource) == params.udpSource);
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_ALLOW_FRAME_DROP, allowFrameDrop) && static_cast<bool>(allowFrameDrop) == params.allowFrameDrop);
    ASSERT_TRUE(reader->get(cv::cudacodec::VideoReaderProps::PROP_RAW_MODE, rawMode) && static_cast<bool>(rawMode) == params.rawMode);
}

CUDA_TEST_P(Seek, Reader)
{
    std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    // seek to a non key frame
    const int firstFrameIdx = 18;

    GpuMat frameGs;
    {
        cv::Ptr<cv::cudacodec::VideoReader> readerGs = cv::cudacodec::createVideoReader(inputFile);
        ASSERT_TRUE(readerGs->set(cudacodec::ColorFormat::GRAY));
        for (int i = 0; i <= firstFrameIdx; i++)
            ASSERT_TRUE(readerGs->nextFrame(frameGs));
    }

    cudacodec::VideoReaderInitParams params;
    params.firstFrameIdx = firstFrameIdx;
    cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile, {}, params);
    double iFrame = 0.;
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, iFrame));
    ASSERT_EQ(iFrame, static_cast<double>(firstFrameIdx));
    ASSERT_TRUE(reader->set(cudacodec::ColorFormat::GRAY));
    GpuMat frame;
    ASSERT_TRUE(reader->nextFrame(frame));
    ASSERT_EQ(cuda::norm(frameGs, frame, NORM_INF), 0.0);
    ASSERT_TRUE(reader->get(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, iFrame));
    ASSERT_EQ(iFrame, static_cast<double>(firstFrameIdx+1));
}

#endif // HAVE_NVCUVID

#if defined(HAVE_NVCUVID) && defined(HAVE_NVCUVENC)
struct TransCode : testing::TestWithParam<cv::cuda::DeviceInfo>
{
    cv::cuda::DeviceInfo devInfo;
    virtual void SetUp()
    {
        devInfo = GetParam();
        cv::cuda::setDevice(devInfo.deviceID());
    }
};

#if defined(WIN32)  // remove when FFmpeg wrapper includes PR25874
#define WIN32_WAIT_FOR_FFMPEG_WRAPPER_UPDATE
#endif

CUDA_TEST_P(TransCode, H264ToH265)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.h264";
    constexpr cv::cudacodec::ColorFormat colorFormat = cv::cudacodec::ColorFormat::NV_NV12;
    constexpr double fps = 25;
    const cudacodec::Codec codec = cudacodec::Codec::HEVC;
    const std::string ext = ".mp4";
    const std::string outputFile = cv::tempfile(ext.c_str());
    constexpr int nFrames = 5;
    Size frameSz;
    {
        cv::Ptr<cv::cudacodec::VideoReader> reader = cv::cudacodec::createVideoReader(inputFile);
        cv::cudacodec::FormatInfo fmt = reader->format();
        reader->set(cudacodec::ColorFormat::NV_NV12);
        cv::Ptr<cv::cudacodec::VideoWriter> writer;
        cv::cuda::GpuMat frame;
        cv::cuda::Stream stream;
        for (int i = 0; i < nFrames; ++i) {
            ASSERT_TRUE(reader->nextFrame(frame, stream));
            ASSERT_FALSE(frame.empty());
            Mat tst; frame.download(tst);
            if (writer.empty()) {
                frameSz = Size(fmt.width, fmt.height);
                writer = cv::cudacodec::createVideoWriter(outputFile, frameSz, codec, fps, colorFormat, 0, stream);
            }
            writer->write(frame);
        }
    }

    {
        cv::VideoCapture cap(outputFile);
        ASSERT_TRUE(cap.isOpened());
        const int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        ASSERT_EQ(frameSz, Size(width, height));
        ASSERT_EQ(fps, cap.get(CAP_PROP_FPS));
        Mat frame;
        for (int i = 0; i < nFrames; ++i) {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
#if !defined(WIN32_WAIT_FOR_FFMPEG_WRAPPER_UPDATE)
            const int pts = static_cast<int>(cap.get(CAP_PROP_PTS));
            ASSERT_EQ(i, pts > 0 ? pts : 0); // FFmpeg back end returns dts if pts is zero.
#endif
        }
    }
    ASSERT_EQ(0, remove(outputFile.c_str()));
}

INSTANTIATE_TEST_CASE_P(CUDA_Codec, TransCode, ALL_DEVICES);

#endif

#if defined(HAVE_NVCUVENC)

>>>>>>> 80f1ca2442982ed518076cd88cf08c71155b30f6
//////////////////////////////////////////////////////
// VideoWriter

CUDA_TEST_P(Video, Writer)
{
    cv::cuda::setDevice(GET_PARAM(0).deviceID());
<<<<<<< HEAD

    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "video/" + GET_PARAM(1);

    std::string outputFile = cv::tempfile(".avi");
    const double FPS = 25.0;

    cv::VideoCapture reader(inputFile);
    ASSERT_TRUE(reader.isOpened());

    cv::Ptr<cv::cudacodec::VideoWriter> d_writer;

    cv::Mat frame;
    cv::cuda::GpuMat d_frame;

    for (int i = 0; i < 10; ++i)
=======
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    const bool deviceSrc = GET_PARAM(1);
    const cudacodec::Codec codec = GET_PARAM(2);
    const double fps = GET_PARAM(3);
    const cv::cudacodec::ColorFormat colorFormat = GET_PARAM(4);
    const std::string ext = ".mp4";
    const std::string outputFile = cv::tempfile(ext.c_str());
    constexpr int nFrames = 5;
    Size frameSz;
>>>>>>> 80f1ca2442982ed518076cd88cf08c71155b30f6
    {
        reader >> frame;
        ASSERT_FALSE(frame.empty());

        d_frame.upload(frame);

        if (d_writer.empty())
            d_writer = cv::cudacodec::createVideoWriter(outputFile, frame.size(), FPS);

        d_writer->write(d_frame);
    }

    reader.release();
    d_writer.release();

    reader.open(outputFile);
    ASSERT_TRUE(reader.isOpened());

    for (int i = 0; i < 5; ++i)
    {
<<<<<<< HEAD
        reader >> frame;
        ASSERT_FALSE(frame.empty());
=======
        cv::VideoCapture cap(outputFile);
        ASSERT_TRUE(cap.isOpened());
        const int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        ASSERT_EQ(frameSz, Size(width, height));
        ASSERT_EQ(fps, cap.get(CAP_PROP_FPS));
        Mat frame;
        for (int i = 0; i < nFrames; ++i) {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
#if !defined(WIN32_WAIT_FOR_FFMPEG_WRAPPER_UPDATE)
            const int pts = static_cast<int>(cap.get(CAP_PROP_PTS));
            ASSERT_EQ(i, pts > 0 ? pts : 0); // FFmpeg back end returns dts if pts is zero.
#endif
        }
>>>>>>> 80f1ca2442982ed518076cd88cf08c71155b30f6
    }
}

<<<<<<< HEAD
#endif // _WIN32, HAVE_NVCUVENC

#define VIDEO_SRC "cv/video/768x576.avi", "cv/video/1920x1080.avi", "highgui/video/big_buck_bunny.avi", \
=======
#define DEVICE_SRC true, false
#define FPS 10, 29
#define CODEC cv::cudacodec::Codec::H264, cv::cudacodec::Codec::HEVC
#define COLOR_FORMAT cv::cudacodec::ColorFormat::BGR, cv::cudacodec::ColorFormat::RGB, cv::cudacodec::ColorFormat::BGRA, \
cv::cudacodec::ColorFormat::RGBA, cv::cudacodec::ColorFormat::GRAY
INSTANTIATE_TEST_CASE_P(CUDA_Codec, Write, testing::Combine(ALL_DEVICES, testing::Values(DEVICE_SRC), testing::Values(CODEC), testing::Values(FPS),
    testing::Values(COLOR_FORMAT)));

PARAM_TEST_CASE(EncoderParams, cv::cuda::DeviceInfo, int)
{
    cv::cuda::DeviceInfo devInfo;
    cv::cudacodec::EncoderParams params;
    virtual void SetUp()
    {
        devInfo = GET_PARAM(0);
        cv::cuda::setDevice(devInfo.deviceID());
        // Fixed params for CBR test
        params.tuningInfo = cv::cudacodec::EncodeTuningInfo::ENC_TUNING_INFO_HIGH_QUALITY;
        params.encodingProfile = cv::cudacodec::EncodeProfile::ENC_H264_PROFILE_MAIN;
        params.rateControlMode = cv::cudacodec::EncodeParamsRcMode::ENC_PARAMS_RC_CBR;
        params.multiPassEncoding = cv::cudacodec::EncodeMultiPass::ENC_TWO_PASS_FULL_RESOLUTION;
        params.averageBitRate = 1000000;
        params.maxBitRate = 0;
        params.targetQuality = 0;
        params.gopLength = 5;
        params.idrPeriod = GET_PARAM(1);
    }
};

CUDA_TEST_P(EncoderParams, Writer)
{
    const std::string inputFile = std::string(cvtest::TS::ptr()->get_data_path()) + "../highgui/video/big_buck_bunny.mp4";
    constexpr double fps = 25.0;
    constexpr cudacodec::Codec codec = cudacodec::Codec::H264;
    const std::string ext = ".mp4";
    const std::string outputFile = cv::tempfile(ext.c_str());
    Size frameSz;
    const int nFrames = max(params.gopLength, params.idrPeriod) + 1;
    {
        cv::VideoCapture reader(inputFile);
        ASSERT_TRUE(reader.isOpened());
        const cv::cudacodec::ColorFormat colorFormat = cv::cudacodec::ColorFormat::BGR;
        cv::Ptr<cv::cudacodec::VideoWriter> writer;
        cv::Mat frame;
        cv::cuda::GpuMat dFrame;
        cv::cuda::Stream stream;
        for (int i = 0; i < nFrames; ++i) {
            reader >> frame;
            ASSERT_FALSE(frame.empty());
            dFrame.upload(frame);
            if (writer.empty()) {
                frameSz = frame.size();
                writer = cv::cudacodec::createVideoWriter(outputFile, frameSz, codec, fps, colorFormat, params, 0, stream);
                cv::cudacodec::EncoderParams paramsOut = writer->getEncoderParams();
                ASSERT_EQ(params, paramsOut);
            }
            writer->write(dFrame);
        }
    }

    {
        cv::VideoCapture cap(outputFile);
        ASSERT_TRUE(cap.isOpened());
        const int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        ASSERT_EQ(frameSz, Size(width, height));
        ASSERT_EQ(fps, cap.get(CAP_PROP_FPS));
        const bool checkFrameType = videoio_registry::hasBackend(CAP_FFMPEG);
        VideoCapture capRaw;
        int idrPeriod = 0;
        if (checkFrameType) {
            capRaw.open(outputFile, CAP_FFMPEG, { CAP_PROP_FORMAT, -1 });
            ASSERT_TRUE(capRaw.isOpened());
            idrPeriod = params.idrPeriod == 0 ? params.gopLength : params.idrPeriod;
        }
        const double frameTypeIAsciiCode = 73.0; // see CAP_PROP_FRAME_TYPE
        Mat frame, frameRaw;
        for (int i = 0; i < nFrames; ++i) {
            cap >> frame;
            ASSERT_FALSE(frame.empty());
            if (checkFrameType) {
                capRaw >> frameRaw;
                ASSERT_FALSE(frameRaw.empty());
                const bool intraFrameReference = cap.get(CAP_PROP_FRAME_TYPE) == frameTypeIAsciiCode;
                const bool intraFrameActual = i % params.gopLength == 0;
                ASSERT_EQ(intraFrameActual, intraFrameReference);
                const bool keyFrameActual = capRaw.get(CAP_PROP_LRF_HAS_KEY_FRAME) == 1.0;
                const bool keyFrameReference = i % idrPeriod == 0;
                ASSERT_EQ(keyFrameActual, keyFrameReference);
#if !defined(WIN32_WAIT_FOR_FFMPEG_WRAPPER_UPDATE)
                const int pts = static_cast<int>(cap.get(CAP_PROP_PTS));
                ASSERT_EQ(i, pts > 0 ? pts : 0); // FFmpeg back end returns dts if pts is zero.
#endif
            }
        }
    }
    ASSERT_EQ(0, remove(outputFile.c_str()));
}

#define IDR_PERIOD testing::Values(5,10)
INSTANTIATE_TEST_CASE_P(CUDA_Codec, EncoderParams, testing::Combine(ALL_DEVICES, IDR_PERIOD));

#endif // HAVE_NVCUVENC

INSTANTIATE_TEST_CASE_P(CUDA_Codec, CheckSet, testing::Combine(
    ALL_DEVICES,
    testing::Values("highgui/video/big_buck_bunny.mp4")));

#define VIDEO_SRC_SCALING "highgui/video/big_buck_bunny.mp4"
#define TARGET_SZ Size2f(1,1), Size2f(0.8f,0.9f), Size2f(2.3f,1.8f)
#define SRC_ROI Rect2f(0,0,1,1), Rect2f(0.25f,0.25f,0.5f,0.5f)
#define TARGET_ROI Rect2f(0,0,1,1), Rect2f(0.2f,0.3f,0.6f,0.7f)
INSTANTIATE_TEST_CASE_P(CUDA_Codec, Scaling, testing::Combine(
    ALL_DEVICES, testing::Values(VIDEO_SRC_SCALING), testing::Values(TARGET_SZ), testing::Values(SRC_ROI), testing::Values(TARGET_ROI)));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, DisplayResolution, ALL_DEVICES);

#define VIDEO_SRC_R testing::Values("highgui/video/big_buck_bunny.mp4", "cv/video/768x576.avi", "cv/video/1920x1080.avi", "highgui/video/big_buck_bunny.avi", \
    "highgui/video/big_buck_bunny.h264", "highgui/video/big_buck_bunny.h265", "highgui/video/big_buck_bunny.mpg", \
    "highgui/video/sample_322x242_15frames.yuv420p.libvpx-vp9.mp4")
    //, "highgui/video/sample_322x242_15frames.yuv420p.libaom-av1.mp4", \
    "cv/tracking/faceocc2/data/faceocc2.webm", "highgui/video/sample_322x242_15frames.yuv420p.mpeg2video.mp4", "highgui/video/sample_322x242_15frames.yuv420p.mjpeg.mp4")

INSTANTIATE_TEST_CASE_P(CUDA_Codec, Video, testing::Combine(ALL_DEVICES,VIDEO_SRC_R));

const color_conversion_params_t color_conversion_params[] =
{
    color_conversion_params_t("highgui/video/big_buck_bunny.h264", false),
    color_conversion_params_t("highgui/video/big_buck_bunny_full_color_range.h264", true),
};

#define VIDEO_COLOR_OUTPUTS cv::cudacodec::ColorFormat::BGRA, cv::cudacodec::ColorFormat::BGRA
INSTANTIATE_TEST_CASE_P(CUDA_Codec, ColorConversion, testing::Combine(
    ALL_DEVICES,
    testing::Values(VIDEO_COLOR_OUTPUTS),
    testing::ValuesIn(color_conversion_params)));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, ReconfigureDecoderWithScaling, ALL_DEVICES);

#define N_DECODE_SURFACES testing::Values(0, 10)
INSTANTIATE_TEST_CASE_P(CUDA_Codec, ReconfigureDecoder, testing::Combine(ALL_DEVICES, N_DECODE_SURFACES));

#define VIDEO_SRC_RW "highgui/video/big_buck_bunny.h264", "highgui/video/big_buck_bunny.h265"
INSTANTIATE_TEST_CASE_P(CUDA_Codec, VideoReadRaw, testing::Combine(
    ALL_DEVICES,
    testing::Values(VIDEO_SRC_RW)));

const histogram_params_t histogram_params[] =
{
    histogram_params_t("highgui/video/big_buck_bunny.mp4", false),
    histogram_params_t("highgui/video/big_buck_bunny.h264", true),
    histogram_params_t("highgui/video/big_buck_bunny_full_color_range.h264", true),
};

INSTANTIATE_TEST_CASE_P(CUDA_Codec, Histogram, testing::Combine(ALL_DEVICES,testing::ValuesIn(histogram_params)));

const check_extra_data_params_t check_extra_data_params[] =
{
    check_extra_data_params_t("highgui/video/big_buck_bunny.mp4", 45),
    check_extra_data_params_t("highgui/video/big_buck_bunny.mov", 45),
    check_extra_data_params_t("highgui/video/big_buck_bunny.mjpg.avi", 0)
};

INSTANTIATE_TEST_CASE_P(CUDA_Codec, CheckExtraData, testing::Combine(
    ALL_DEVICES,
    testing::ValuesIn(check_extra_data_params)));

#define VIDEO_SRC_KEY "highgui/video/big_buck_bunny.mp4", "cv/video/768x576.avi", "cv/video/1920x1080.avi", "highgui/video/big_buck_bunny.avi", \
>>>>>>> 80f1ca2442982ed518076cd88cf08c71155b30f6
    "highgui/video/big_buck_bunny.h264", "highgui/video/big_buck_bunny.h265", "highgui/video/big_buck_bunny.mpg"
INSTANTIATE_TEST_CASE_P(CUDA_Codec, Video, testing::Combine(
    ALL_DEVICES,
    testing::Values(VIDEO_SRC)));

INSTANTIATE_TEST_CASE_P(CUDA_Codec, Seek, ALL_DEVICES);

#endif // HAVE_NVCUVID || HAVE_NVCUVENC
}} // namespace
