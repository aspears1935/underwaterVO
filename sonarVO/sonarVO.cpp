/*
 * File: sonarVO.cpp
 * Name: Anthony Spears
 * Date: April 2, 2016
 * Description: Rotation and Translation Estimation with Sonar Data for Visual Odometry
 */


#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#include <bvt_sdk.h>

#include <cv.h>
//#include <highgui.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "RANSAC.hpp"
#include "my_ptsetreg.hpp"
#include "my_five-point.hpp"

using namespace std;
using namespace cv;

#define DISPLAY_IMGS 	true
#define STOP_BETWEEN_IMGS false
#define VERBOSE   false
#define WRITE_FRAME_NUM false
#define BLUR_SON_IMGS   true
#define WRITE_IMGS      false
#define THRESHOLD_SON_IMGS false
#define APPLY_CLAHE false
#define SONAR_CARTESIAN true //Cartesian (x,y) vs polar (range,bearing)
//#define VIDEO_SONAR_FILE true //Use a video file instead of sonar file
#define PROCESS_00_FRAME false
#define USE_RADIUS_MATCH false
#define THRESHOLD_JUMPS true
#define PRINT_RAW_SHIFTS false
#define WRITE_VIDEO true

#define MAX_X 10 //10 Pixels
#define MAX_Y 20 //20 Pixels
#define MAX_PSI 20 //20 degrees
#define MAPSAC 7
#define ROBUST_EST_METH CV_FM_LMEDS //CV_FM_LMEDS or CV_FM_RANSAC or MAPSAC

#define PI 3.14159265359
#define ZERO 0.0001
#define max_BINARY_value 255
#define FOV_SONAR 45 //P900-45 has a 45 degree FOV. X direction in Opencv.

// RNG rng(12345); //Not needed currently

 //Global Variable to save trackbar frame number
 bool tbar_update = false; 
 bool VIDEO_SONAR_FILE = false; //Use a video file instead of sonar file

//-----------------Trackbar Handler---------------//
 void Trackbar(int, void*)
 {
   tbar_update = true;
 }

//------------------Main Function----------------//
 int main( int argc, char *argv[] )
 {

   int i = 0;
   bool pause = false; 
   char keypushed = 0;

   double xinit = 0;
   double yinit = 0;
   double yawinit = 0;
   
   double sonOFsum[3] = {0,0,0};
   double sonSURFsum[3] = {0,0,0};
   double sonSIFTsum[3] = {0,0,0};
   double sonHARRISsum[3] = {0,0,0};

   Size sonarSizeLast;
   double SONAR_TIME0 = 0;
   double ping_time_sec = 0;  //Time of Ping from TIME0 in Seconds
   double prev_ping_time_sec = 0;  //Time of Ping from TIME0 in Seconds
   double VIDEO_FPS = 0;

   int OF_EstValid = -1;
   int SURF_EstValid = -1;
   int SIFT_EstValid = -1;
   int HARRIS_EstValid = -1;
   
   //Print out OPENCV Version:
   cout << "Using OpenCV v" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << endl;
   
   if (argc < 2) {
     printf("usage: ./sonar_opencv <sonar-file> <OPTIONAL xinit> <OPTIONAL yinit> <OPTIONAL yawinit degrees>\n");
     printf("example: ./sonar_opencv ../../../data/swimmer.son 0 0 -90\n");
     exit(-1);
   }

   //If global initial values, read them in
   if (argc > 2) 
     xinit = atof(argv[2]);
   if (argc > 3)
     yinit = atof(argv[3]);
   if (argc > 4)
     yawinit = atof(argv[4]);

   cout << "Initial Global Values: x,y,yaw = " << xinit << "," << yinit << "," << yawinit << endl;

   double sonOFsumGlobal[3] = {xinit,yinit,yawinit};
   double sonSURFsumGlobal[3] = {xinit,yinit,yawinit};
   double sonSIFTsumGlobal[3] = {xinit,yinit,yawinit};
   double sonHARRISsumGlobal[3] = {xinit,yinit,yawinit};
   
   char SonarFile[256];
   strcpy(SonarFile, argv[1]);
   
      //Find out if input is a son file (real data) or an avi video file (sim data)
      cout << SonarFile[strlen(SonarFile)-3] << SonarFile[strlen(SonarFile)-2] << SonarFile[strlen(SonarFile)-1] << endl;
      if(((SonarFile[strlen(SonarFile)-3]=='s')||(SonarFile[strlen(SonarFile)-3]=='S'))
	 &&((SonarFile[strlen(SonarFile)-2]=='o')||(SonarFile[strlen(SonarFile)-2]=='O'))
	 &&((SonarFile[strlen(SonarFile)-1]=='n')||(SonarFile[strlen(SonarFile)-1]=='N')))
	{
	  VIDEO_SONAR_FILE = false;
	  cout << "Found .SON file input name" << endl;
	}
      else if(((SonarFile[strlen(SonarFile)-3]=='a')||(SonarFile[strlen(SonarFile)-3]=='A'))
	      &&((SonarFile[strlen(SonarFile)-2]=='v')||(SonarFile[strlen(SonarFile)-2]=='V'))
	      &&((SonarFile[strlen(SonarFile)-1]=='i')||(SonarFile[strlen(SonarFile)-1]=='I')))
	{
	  VIDEO_SONAR_FILE = true;
	  cout << "Found .AVI file input name" << endl;
	}
      else if(((SonarFile[strlen(SonarFile)-3]=='m')||(SonarFile[strlen(SonarFile)-3]=='M'))
	      &&((SonarFile[strlen(SonarFile)-2]=='p')||(SonarFile[strlen(SonarFile)-2]=='P'))
	      &&((SonarFile[strlen(SonarFile)-1]=='4')||(SonarFile[strlen(SonarFile)-1]=='4')))
	{
	  VIDEO_SONAR_FILE = true;
	  cout << "Found .MP4 file input name" << endl;
	}
      else
	{
	  cout << "ERROR: Unsupported file type." << endl;
	  return -1;
	}

      int ret;
      int pings = -1;
      int height, width;
      BVTHead head = NULL;
      BVTSonar son = BVTSonar_Create();
      BVTColorMapper mapper;
      VideoCapture inputSonarVideo; //      VideoCapture inputSonarVideo(SonarFile);
      if(VIDEO_SONAR_FILE)
	inputSonarVideo = VideoCapture(SonarFile);

      //Create Windows:
      char color_wnd[] = "Sonar: 'b'=back, 'f'=forward, 'p'=pause, 'ESC'=exit";
      char son_features_wnd[] = "Sonar Features Window";
      char son_matches_wnd[] = "Sonar Matches Window";

      if(DISPLAY_IMGS)
	{
	  namedWindow(color_wnd, 1);	
	  namedWindow(son_features_wnd, 1);	
	  namedWindow(son_matches_wnd, 1);
	}

      Mat sonarImgGray, sonFeaturesImg, sonMatchesImg;
      Mat sonMatchesImgOF;

      if(!VIDEO_SONAR_FILE) //Using .son file
	{
	  // Create a new BVTSonar Object
	  if( son == NULL )
	    {
	      printf("BVTSonar_Create: failed\n");
	      return 1;
	    }
	  // Open the sonar
	  ret = BVTSonar_Open(son, "FILE", SonarFile);
	  if( ret != 0 )
	    {
	      printf("BVTSonar_Open: ret=%d\n", ret);
	      return 1;
	    }
	  
	  // Make sure we have the right number of heads
	  int heads = -1;
	  heads = BVTSonar_GetHeadCount(son);
	  printf("BVTSonar_GetHeadCount: %d\n", heads);
	  
	  // Get the first head
	  head = NULL;
	  ret = BVTSonar_GetHead(son, 0, &head);
	  if( ret != 0 )
	    {
	      printf("BVTSonar_GetHead: ret=%d\n", ret);
	      return 1;
	    }
	
	  // Check the ping count
	  pings = -1;
	  pings = BVTHead_GetPingCount(head);
	  printf("BVTHead_GetPingCount: %d\n", pings);
	  
	  // Set the range window
	  BVTHead_SetRange(head, 0, 50);

	  // Build a color mapper
	  mapper = BVTColorMapper_Create();
	  if( mapper == NULL )
	    {
	      printf("BVTColorMapper_Create: failed\n");
	      return 1;
	    }
	  
	  // Load the bone colormap
	  ret = BVTColorMapper_Load(mapper, "/home/anthony/sonar-processing/bvtsdk/colormaps/bone.cmap");
	  if( ret != 0 )
	    {
	      printf("BVTColorMapper_Load: ret=%d\n", ret);
	      return 1;
	    }
	  
	  cout << "AUTOTHRESH = " << BVTColorMapper_GetAutoMode(mapper) << endl;
	  BVTColorMapper_SetAutoMode(mapper, 0);
	  cout << "Changed. AUTOTHRESH = " << BVTColorMapper_GetAutoMode(mapper) << endl;
	  cout << "Top Threshold = " << BVTColorMapper_GetTopThreshold(mapper) << endl;
	  cout << "Bottom Threshold = " << BVTColorMapper_GetBottomThreshold(mapper) << endl;
	}

      else //Use video file instead of .son file
	{
	  if(!inputSonarVideo.isOpened())
	    {
	      cout << "CANNOT OPEN INPUT VIDEO: " << SonarFile << endl;
	      return -1;
	    }
	  pings = inputSonarVideo.get(CAP_PROP_FRAME_COUNT);//Get number of frames
	}

      //------------------Get File Information--------------------//
      //----------------------------------------------------------//
      if(!VIDEO_SONAR_FILE) //If Using .son file
	{
	  BVTPing ping = NULL;
	  ret = BVTHead_GetPing(head, 0, &ping);
	  if( ret != 0 )
	    {
	      printf("BVTHead_GetPing: ret=%d\n", ret);
	      return 1;
	    }

	  //Get Ping0 Timestamp for TIME0
	  SONAR_TIME0 = BVTPing_GetTimestamp(ping);

	  BVTMagImage bvtmag_img;	//Magnitude Image
	  BVTColorImage bvtclr_img;	//Color Mapped Image
	  
	  // Generate an image from the ping:	  
	  if(SONAR_CARTESIAN)
	    ret = BVTPing_GetImageXY(ping, &bvtmag_img);
	  else //POLAR (range,bearing)
	    ret = BVTPing_GetImageRTheta(ping, &bvtmag_img);
	  if( ret != 0 )
	    {
	      printf("BVTPing_GetImage: ret=%d\n", ret);
	      return 1;
	    }
	  
	  height = BVTMagImage_GetHeight(bvtmag_img);
	  width = BVTMagImage_GetWidth(bvtmag_img);
	  /*double rangeRes = BVTMagImage_GetRangeResolution(bvtmag_img);
	  double bearingRes = BVTMagImage_GetBearingResolution(bvtmag_img);
	  cout << "Range Resolution MAG " << rangeRes << " meters/pixel" << endl;
	  cout << "Bearing Resolution MAG " << bearingRes << " deg/pixel" << endl;
	  */
	  BVTPing_Destroy(ping);
	  cout << "SON Data!" << endl << "Input frame resolution: Width=" << width << "  Height=" << height << " Num of Pings=" << pings << endl;
	}
      else //Video instead of .son file
	{
	  //Get Video Properties
	  width = (int) inputSonarVideo.get(CAP_PROP_FRAME_WIDTH);
	  height = (int) inputSonarVideo.get(CAP_PROP_FRAME_HEIGHT);
	  VIDEO_FPS = inputSonarVideo.get(CAP_PROP_FPS);
	  cout << "Video Data!" << endl << "Input frame resolution: Width=" << width << "  Height=" << height << " Num of Pings=" << pings << " FPS=" << VIDEO_FPS << std::endl;
	}
      sonarSizeLast = Size(width,height);
      
      
      //Create Trackbars:             
      int max_frame_num = pings - 1;
      char position_tbar[] = "Position (0 based)";
      if(DISPLAY_IMGS)
	createTrackbar(position_tbar, color_wnd, &i, max_frame_num, Trackbar);

      //-----------------------------------------------------------
      //------------------- Initialize Feature Detection ----------
      //-----------------------------------------------------------
      //SURF Detector:
      int minHessian = 300; //Suggested 300-500. Higher val gives less features.
      SurfFeatureDetector sonSURFdetector(minHessian);
      SurfDescriptorExtractor sonSURFextractor;
      FlannBasedMatcher sonSURFmatcher;
      
      //SIFT Detector:
      double SIFTthreshold = 0.001; //0.04
      double SIFTedgeThreshold = 100; //10 -> larger = more features
      cv::Ptr< cv::FeatureDetector > sonSIFTdetector = FeatureDetector::create("SIFT");
      cv::Ptr< cv::DescriptorExtractor > sonSIFTextractor = DescriptorExtractor::create("SIFT");
      sonSIFTdetector->set("contrastThreshold", SIFTthreshold);
      sonSIFTdetector->set("edgeThreshold", SIFTedgeThreshold);
      sonSIFTdetector->set("nFeatures", 1000);
      sonSIFTdetector->set("nOctaveLayers", 5);
      sonSIFTdetector->set("sigma", 1.0);//1.6 lower seems to be better to a point
      FlannBasedMatcher sonSIFTmatcher;

      //DEBUG: Print out detector properties
      /*	std::vector<cv::String> parameters;
	vidSIFTdetector->getParams(parameters);
	for(int i2 = 0; i2 < parameters.size(); i2++)
	{
	cout <<  parameters[i2] << endl;
	}
	for(;;);*/
      //End DEBUG Print out ...

      //DEBUG: SIFT creation with no Common Interface
      /*SiftFeatureDetector sonSIFTdetector;//SIFTthreshold, SIFTedgeThreshold);
	SiftDescriptorExtractor sonSIFTextractor;
	FlannBasedMatcher sonSIFTmatcher;	
	SiftFeatureDetector vidSIFTdetector(SIFTthreshold, SIFTedgeThreshold);
      	SiftDescriptorExtractor vidSIFTextractor;
	FlannBasedMatcher vidSIFTmatcher;
      */

      //HARRIS Detector:
      cv::Ptr<cv::FeatureDetector> sonHARRISdetector = cv::FeatureDetector::create("HARRIS"); 
	// if you want it faster take e.g. ORB or FREAK
      cv::Ptr<cv::DescriptorExtractor> sonHARRISdescriptor = cv::DescriptorExtractor::create("SIFT"); 
      sonHARRISdetector->set("nfeatures", 1000);
      //sonHARRISdetector->set("minDistance", 10);                                 
      sonHARRISdetector->set("qualityLevel", 0.001); //0.01  
      FlannBasedMatcher sonHARRISmatcher;  

      //Create CLAHE:
      int CLAHEclipLimit = 4;//was 6
      cv::Size CLAHEtileGridSize(16,16);
      Ptr<CLAHE> clahe = createCLAHE(CLAHEclipLimit, CLAHEtileGridSize);
      clahe->setClipLimit(CLAHEclipLimit); //Don't set limit to just use default?

      //-------Open Output File, write column headings and run properties-------//
      //------------------------------------------------------------------------//
      ofstream outfileSonar("outputSonar.csv");
      outfileSonar << "PROPERTIES:;";
      outfileSonar << "SonarFileName=" << SonarFile << ";";
      outfileSonar << fixed;
      outfileSonar << setprecision(3);
      outfileSonar << "TIME0(sec)=" << SONAR_TIME0 << ";";
      outfileSonar << "numPings=" << pings << ";";
      outfileSonar << "SonarWidth=" << width << ";";
      outfileSonar << "SonarHeight=" << height << ";";
      //ADD MORE!!!!!!!!!!!!!
      if(VIDEO_SONAR_FILE)
	outfileSonar << "FPS=" << VIDEO_FPS << endl;
      outfileSonar << "SURFminHessian=" << minHessian << ";";
      outfileSonar << "SIFTthreshold=" << SIFTthreshold << ";";
      outfileSonar << "SIFTedgeThresh=" << SIFTedgeThreshold << ";";
      outfileSonar << "CLAHEclipLimit=" << CLAHEclipLimit << ";";
      outfileSonar << "CLAHEtileSize=" << CLAHEtileGridSize << ";";
      outfileSonar << "FOVX=" << FOV_SONAR << ";";
      outfileSonar << "BLUR_IMGS=" << BLUR_SON_IMGS << ";";
      outfileSonar << "ROBUST_EST=" << ROBUST_EST_METH << ";";
      outfileSonar << "PROCESS_00FRAME=" << PROCESS_00_FRAME << ";";
      outfileSonar << "Threshold Imgs=" << THRESHOLD_SON_IMGS << ";";
      outfileSonar << "APPLY_CLAHE=" << APPLY_CLAHE << ";";
      outfileSonar << "CARTESIAN=" << SONAR_CARTESIAN << ";";
      outfileSonar << "VIDEO_SONAR_FILE=" << VIDEO_SONAR_FILE << ";";
      outfileSonar << "PROCESS_00FRAME=" << PROCESS_00_FRAME << ";";
      outfileSonar << "USE_RADIUS_MATCH=" << USE_RADIUS_MATCH << ";";
      outfileSonar << "THRESHOLD_JUMPS=" << THRESHOLD_JUMPS << ";";
      outfileSonar << "MAX_X=" << MAX_X << ";";
      outfileSonar << "MAX_Y=" << MAX_Y << ";";
      outfileSonar << "MAX_PSI=" << MAX_PSI << ";";
      outfileSonar << endl << endl;
      outfileSonar << "time2(sec);frame1;frame2;";
      if(PRINT_RAW_SHIFTS)
	{
	  outfileSonar << "sonOFxshift;sonOFyshift;sonOFtheta;";
	  outfileSonar << "sonSURFxshift;sonSURFyshift;sonSURFtheta;";
	  outfileSonar << "sonSIFTxshift;sonSIFTyshift;sonSIFTtheta;";
	  outfileSonar << "sonHARRISxshift;sonHARRISyshift;sonHARRIStheta;";
	}
      outfileSonar << "sonOFxshift;sonOFyshift;sonOFtheta;sonOFxshiftsum;sonOFyshiftsum;sonOFthetasum;sonOFxshiftsumGlobal;sonOFyshiftsumGlobal;sonOFthetasumGlobal;";
      outfileSonar << "sonSURFxshift;sonSURFyshift;sonSURFtheta;sonSURFxshiftsum;sonSURFyshiftsum;sonSURFthetasum;sonSURFxshiftsumGlobal;sonSURFyshiftsumGlobal;sonSURFthetasumGlobal;";
      outfileSonar << "sonSIFTxshift;sonSIFTyshift;sonSIFTtheta;sonSIFTxshiftsum;sonSIFTyshiftsum;sonSIFTthetasum;sonSIFTxshiftsumGlobal;sonSIFTyshiftsumGlobal;sonSIFTthetasumGlobal;";
      outfileSonar << "sonHARRISxshift;sonHARRISyshift;sonHARRIStheta;sonHARRISxshiftsum;sonHARRISyshiftsum;sonHARRISthetasum;sonHARRISxshiftsumGlobal;sonHARRISyshiftsumGlobal;sonHARRISthetasumGlobal;";
      outfileSonar << "OFnumCorners;OFnumMatches;OFnumInliers;SURFnumCorners;SURFnumMatches;SURFnumInliers;SIFTnumCorners;SIFTnumMatches;SIFTnumInliers;HARRISnumCorners;HARRISnumMatches;HARRISnumInliers;";
      outfileSonar << "OF_EstValid;SURF_EstValid;SIFT_EstValid;HARRIS_EstValid;" << endl;

      //-------Open Out File for GTSAM, write column headings ------------------//
      //------------------------------------------------------------------------//
      ofstream outfileSonarGTSAM("outputSonarGTSAM.csv");
      outfileSonarGTSAM << "length=" << ";" << pings-1 << ";" << endl; // First line should just be length
      outfileSonarGTSAM << "t1;t2;x;y;yaw;numCorners;numMatches;numInliers;estValid;" << endl; // Second line is headings
      
      //---------------------------------------------------------------------
      //----------------Create Video Writer-----------------------------------
      
      VideoWriter outputVideo;
      int outputFPS = 10;
      if(WRITE_VIDEO)
	{
          outputVideo.open("output.avi",cv::VideoWriter::fourcc('X','V','I','D'), outputFPS, Size(width*2,height*2),true);
          if(!outputVideo.isOpened())
            {
              cout << "Could not open the output video for write" << endl;
              return -1;
            }
	}

      //---------------------------------------------------------------------
      //----------------Main Loop--------------------------------------------

      //Create Previous Image Containers:
      Mat prev_sonarImgGray; 
      int prev_i = 0;
      Mat colorImg;

     for (i = 0; i < pings; i++) {
          cout << i+1 << "/" << pings << endl;

	  // Now, get a ping!
	  BVTPing ping = NULL;
	
	  if(!VIDEO_SONAR_FILE) //If Using .son file
	    {
	      ret = BVTHead_GetPing(head, i, &ping);
	      //ret = BVTHead_GetPing(head, -1, &ping);//DEBUG - gets the next one
	      if( ret != 0 )
		{
		  printf("BVTHead_GetPing: ret=%d\n", ret);
		  return 1;
		}
	      
	      // Generate an image from the ping:	  
	      BVTMagImage bvtmag_img;	//Magnitude Image
	      if(SONAR_CARTESIAN)
		ret = BVTPing_GetImageXY(ping, &bvtmag_img);
	      else //POLAR (range,bearing)
		ret = BVTPing_GetImageRTheta(ping, &bvtmag_img);
	      if( ret != 0 )
		{
		  printf("BVTPing_GetImage: ret=%d\n", ret);
		  return 1;
		}
	      
	      //Get Magnitude Image: //Don't Need Now
	      /*height_mag = BVTMagImage_GetHeight(bvtmag_img);
	      width_mag = BVTMagImage_GetWidth(bvtmag_img);
	      IplImage* mag_img_ipl = cvCreateImageHeader(cvSize(width_mag,height_mag), IPL_DEPTH_8U, 1);
	      //cvSetImageData(mag_img_ipl, BVTMagImage_GetBits(bvtmag_img), width_mag);
	      cvReleaseImageHeader(&mag_img_ipl); */ //Create Mag Image

	      BVTColorImage bvtclr_img;	//Color Mapped Image

	      // Perform the colormapping
	      ret = BVTColorMapper_MapImage(mapper, bvtmag_img, &bvtclr_img);
	      if( ret != 0 )
		{
		  printf("BVTColorMapper_MapImage: ret=%d\n", ret);
		  return 1;
		}
	      
	      // Use OpenCV to display the image
	      height = BVTColorImage_GetHeight(bvtclr_img);
	      width = BVTColorImage_GetWidth(bvtclr_img);

	      //Get Range Resolution
	      /*	      double rangeRes = BVTColorImage_GetRangeResolution(bvtclr_img);
	      cout << "Range Resolution: " << rangeRes << " meters/pixel" << endl;
	      */
	      // Create a IplImage header
	      IplImage* color_img_ipl = cvCreateImageHeader(cvSize(width,height), IPL_DEPTH_8U, 4);
	      
	      // And set it's data
	      cvSetImageData(color_img_ipl,  BVTColorImage_GetBits(bvtclr_img), width*4);

	      //Convert IPL image to Mat:
	      //cv::Mat colorImg(color_img_ipl); //Other way of conversion. Bad?
	      colorImg = cvarrToMat(color_img_ipl,true);
	      
	      //Delete ipl image header and bvt imgs
	      cvReleaseImageHeader(&color_img_ipl);
	      BVTColorImage_Destroy(bvtclr_img);
	      BVTMagImage_Destroy(bvtmag_img);

	      //Get Ping Timestamp and find time from Ping0
	      ping_time_sec = BVTPing_GetTimestamp(ping);
	      
	      if(VERBOSE)
		cout << i << " === " << ping_time_sec - SONAR_TIME0 << " sec " << endl;
	    }
	  else //Use video instead of .son file
	    {
	      inputSonarVideo.set(CAP_PROP_POS_FRAMES,i);
	      inputSonarVideo >> colorImg;
	      height = colorImg.rows;
	      width = colorImg.cols;
	      
	      //Get Ping Time
	      ping_time_sec = i*(1/VIDEO_FPS);
	      
	      if(VERBOSE)
		cout << i << " === " << ping_time_sec << " sec " << endl;
	    }

	  Size sonarSize(width,height);
	  if(VERBOSE)
	    cout << "Sonar Size = " << sonarSize << endl;


	//------------------------------------------------------------
	//----------- Preprocess SONAR Frame -------------------------
	//------------------------------------------------------------

	//Get Gray Sonar Image:
	cvtColor(colorImg, sonarImgGray, COLOR_BGR2GRAY);

	//DEBUG:
	//Remove Lines (cols 112-114 and 142-144) from Sonar Images:
	/*	sonarImgGray.col(111).copyTo(sonarImgGray.col(112));
	line(sonarImgGray,Point(113,0),Point(113,sonarImgGray.rows-1),Scalar(0,0,0)); //Set the middle column to zero first ...
	
	sonarImgGray.col(113) += sonarImgGray.col(111)*0.5; // ... then add half of left col
	sonarImgGray.col(113) += sonarImgGray.col(115)*0.5; // ... then add half of left col to finally create an average of the two
	sonarImgGray.col(115).copyTo(sonarImgGray.col(114));

	sonarImgGray.col(141).copyTo(sonarImgGray.col(142));
	line(sonarImgGray,Point(143,0),Point(143,sonarImgGray.rows-1),Scalar(0,0,0)); //Set the middle column to zero first ...
        sonarImgGray.col(143) += sonarImgGray.col(141)*0.5; // ... then add half of left col                                                                            
	sonarImgGray.col(143) += sonarImgGray.col(145)*0.5; // ... then add half of left col to finally create an average of the two     
	sonarImgGray.col(145).copyTo(sonarImgGray.col(144));
	*/
	//END DEBUG

	if(WRITE_IMGS)
	  imwrite("SonarGrayIMG.jpg", sonarImgGray);

	//Histogram Equalization with CLAHE:
	if(APPLY_CLAHE)
	  clahe->apply(sonarImgGray,sonarImgGray); //Doesn't help? 

	if(WRITE_IMGS)
	  imwrite("SonarCLAHEIMG.jpg", sonarImgGray);

	//Apply Sonar Image Blurring:
	Size blurKernelSize = Size(5,5);
	if(BLUR_SON_IMGS)
	  GaussianBlur(sonarImgGray,sonarImgGray,blurKernelSize,0,0); //Gaussian BlurOB
	//blur(sonarImgGray,sonarImgGray,blurKernelSize,Point(-1,-1)); //Box Filter Blur
	/*
	//Debug create a known image:
	sonarImgGray = Scalar(0,0,0);
	rectangle(sonarImgGray,Point(100+i,100+i),Point(200+i,200+i),Scalar(255,255,255));*/

	if(WRITE_IMGS)
	  imwrite("SonarBlurIMG.jpg", sonarImgGray);

	//Apply Thresholding:
	int son_thresh_level = 40;
	//int son_thresh_level = 160; //For CLAHE images
	int son_thresh_type = 3;
	if(THRESHOLD_SON_IMGS)
	  cv::threshold(sonarImgGray,sonarImgGray,son_thresh_level,max_BINARY_value,son_thresh_type);

	/*	
	//Higher threshold for middle and edge columns
	int higher_thresh_level = son_thresh_level + 0;
	Mat sonar_roi(sonarImgGray,Rect(112,0,30,sonarImgGray.rows));
	cv::threshold(sonar_roi,sonar_roi,higher_thresh_level,max_BINARY_value,son_thresh_type);
	Mat sonar_roi1(sonarImgGray,Rect(0,0,30,sonarImgGray.rows));
	cv::threshold(sonar_roi1,sonar_roi1,higher_thresh_level,max_BINARY_value,son_thresh_type);
	Mat sonar_roi2(sonarImgGray,Rect(225,0,30,sonarImgGray.rows));
	cv::threshold(sonar_roi2,sonar_roi2,higher_thresh_level,max_BINARY_value,son_thresh_type);
	*/

	//Fill in previous if i = 0:
	if(i == 0)
	  {
	    cout << "I is 0 - saving fake previous image" << endl;
	    prev_sonarImgGray = sonarImgGray.clone();
	    prev_i = i;
	    prev_ping_time_sec = ping_time_sec;
	    if(!PROCESS_00_FRAME)
	      continue; //Don't process 0-0 frame. Go to 0-1 frame.
	  }

	//Check if different size from last frame (resized), then skip frame
	if((sonarSize.width != sonarSizeLast.width)||(sonarSize.height != sonarSizeLast.height))
	  {
	    cout << "FRAMES ARE DIFFERENT SIZES! SKIPPING." << endl;
	    cout << sonarSizeLast << sonarSize << endl;
	    sonarSizeLast = Size(sonarSize.width,sonarSize.height);

	    //Save new prev image
	    prev_sonarImgGray = sonarImgGray.clone();
	    prev_i = i;
	    prev_ping_time_sec = ping_time_sec;
	    BVTPing_Destroy(ping);
	    continue;
	  }
	
	//-------------------------------------------------------------------
	//----------- Get SONAR Shifts --------------------------------------
	//-------------------------------------------------------------------

	//------------------ Sparse Optical Flow Estimates ---------------//
	// Parameters for Shi-Tomasi algorithm                                      
	vector<Point2f> sonOFcorners1;
	vector<Point2f> sonOFcorners2;
	vector<Point2f> sonOFpts_diff;
	vector<KeyPoint> sonOFkeypoints1;
	vector<KeyPoint> sonOFkeypoints2;
        double sonOFqualityLevel = 0.01;
	double sonOFminDistance = 10;
	int sonOFblockSize = 3;
        bool sonOFuseHarrisDetector = false;
        double sonOF_k = 0.04;
        int sonOF_r = 3;      //Radius of points for Corners 
	TermCriteria sonOFtermcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
        Size sonOFwinSize(31,31);
	int sonOFmax_corners = 1000;
	Mat sonOFmodel, sonOFmask;
	int sonOFnumInliers = 0;
	int sonOFnumCorners = 0;
	int sonOFnumMatches = 0;

	//Corner Detection:
	goodFeaturesToTrack( prev_sonarImgGray, sonOFcorners1, sonOFmax_corners, sonOFqualityLevel, sonOFminDistance, Mat(), sonOFblockSize, sonOFuseHarrisDetector, sonOF_k);

	sonOFnumCorners = sonOFcorners1.size();

        if(sonOFcorners1.size() > 0) //Detected some corners
	  {
	    //Calculate Corners Subpixel Accuracy:
	    Size sonOFsubPixWinSize(10,10);
	    cornerSubPix(prev_sonarImgGray, sonOFcorners1, sonOFsubPixWinSize, Size(-1,-1), sonOFtermcrit);

	    //Lucas Kanade Pyramid Algorithm:        
	    vector<uchar> sonOFstatus;
	    vector<float> sonOFerr;
	    calcOpticalFlowPyrLK(prev_sonarImgGray, sonarImgGray, sonOFcorners1, sonOFcorners2, sonOFstatus, sonOFerr, sonOFwinSize, 7, sonOFtermcrit, 0, 0.001);

	    if(VERBOSE)
	      cout << "NUM OF CORNERS:" << sonOFcorners1.size() << endl;

	    //std::vector< DMatch > sonOFgood_matches;//DEBUG - not used
	    std::vector<Point2f> sonOFgood_match_pts1;
	    std::vector<Point2f> sonOFgood_match_pts2;

	    for(int i1=0; i1 < sonOFcorners1.size(); i1++)
	      {
		if(sonOFstatus[i1])
		  {
		    sonOFpts_diff.push_back(sonOFcorners2[i1] - sonOFcorners1[i1]);
		    //DEBUG: Print all points:
		    //cout << "corner1: " << sonOFcorners1[i1] << " -- corner2:" << sonOFcorners2[i1] << endl;
		    //int i_diff = sonOFpts_diff.size()-1;
		    //cout << i_diff << ": " << sonOFpts_diff[i_diff] << endl;
		    //END DEBUG: Print all points

		    //Save Points into Keypoint Form For Drawing Matches:
		    sonOFkeypoints1.push_back(KeyPoint(sonOFcorners1[i1],1.f));
		    sonOFkeypoints2.push_back(KeyPoint(sonOFcorners2[i1],1.f));  
		    sonOFgood_match_pts1.push_back( sonOFcorners1[i1] );
		    sonOFgood_match_pts2.push_back( sonOFcorners2[i1] );
		    //float tmpDist = sqrt(sonOFpts_diff[i_diff].x*sonOFpts_diff[i_diff].x + sonOFpts_diff[i_diff].y*sonOFpts_diff[i_diff].y); //Calculate distance between
		  }
	      }

	    sonOFnumMatches = sonOFpts_diff.size();

	if(sonOFnumMatches >= 3) //If found enough matches
	  {
	    //Run RANSAC                                                        
	    double sonOF_RANSAC_reprojthresh = 1;
	    double sonOF_RANSAC_param = 0.99;
	    int sonOFok;
	    if(SONAR_CARTESIAN)
	      sonOFok = estimateRigidTransform2DNew(sonOFgood_match_pts1,sonOFgood_match_pts2, ROBUST_EST_METH, sonOFmodel, sonOFmask, sonOF_RANSAC_reprojthresh, sonOF_RANSAC_param, sonarSize);
	    else //POLAR
	      sonOFok = estimateTranslationNew(sonOFpts_diff,sonOFpts_diff, ROBUST_EST_METH, sonOFmodel, sonOFmask, sonOF_RANSAC_reprojthresh, sonOF_RANSAC_param);

	    sonOFnumInliers = sum(sonOFmask)[0]; //Find number of inliers
	    
	    OF_EstValid = 1; // Valid data flag true
	    
	    if(DISPLAY_IMGS)
	      {
		//------------------Show Features----------------------// 
		//Get copy of gray img to mark features:         
		cvtColor(prev_sonarImgGray, sonFeaturesImg, CV_GRAY2BGR);

		//DEBUG: Other way of drawing matches. Doesn't work?
		//drawMatches(prev_sonarImgGray, sonOFkeypoints1, sonarImgGray, sonOFkeypoints2, sonOFgood_matches, matchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		
		//Create Matches Image:
		Mat tmpMatchesImg = Mat(sonarImgGray.rows,2*sonarImgGray.cols,CV_8U);
		Point2f sonOFimg2offset = Point2f(sonarImgGray.cols,0);
		Mat son_roi1(tmpMatchesImg,Rect(0,0,sonarImgGray.cols,sonarImgGray.rows));
		prev_sonarImgGray.copyTo(son_roi1);
		Mat son_roi2(tmpMatchesImg,Rect(sonarImgGray.cols,0,sonarImgGray.cols,sonarImgGray.rows));
		sonarImgGray.copyTo(son_roi2);
		cvtColor(tmpMatchesImg, sonMatchesImg, CV_GRAY2BGR);  
		
		//Draw Inliers and matches:
		for(int i1 = 0; i1 < sonOFnumMatches; i1++)
		  {
		    Scalar color;
		    if(sonOFmask.at<char>(0,i1))
		      color = Scalar(0,255,0); //Green
		    else
		      color = Scalar(0,0,255); //Red
		    
		    //Draw features:
		    circle(sonFeaturesImg, sonOFcorners1[i1], 4, color);

		    //Draw Matches and lines between:
		    circle(sonMatchesImg, sonOFkeypoints1[i1].pt, 4, color);
		    circle(sonMatchesImg, sonOFkeypoints2[i1].pt + sonOFimg2offset, 4, color);
		    line(sonMatchesImg, sonOFkeypoints1[i1].pt, sonOFkeypoints2[i1].pt+sonOFimg2offset, color);
		  }

		sonMatchesImgOF = sonMatchesImg.clone();
	      }
	  }
	else //Didn't find enough matches
	  {
	    cout << "NO OF MODEL FOUND" << endl;
	    sonOFmodel = Mat::zeros(1,3,CV_64F);
	    OF_EstValid = 0; // Valid data flag
	  }
	
	  }
	else //No corners found:
	  {
	    cout << "NO OF MODEL FOUND" << endl;
	    sonOFmodel = Mat::zeros(1,3,CV_64F);
	    OF_EstValid = 0; // Valid data flag
	  }

	if(VERBOSE)
	  {
	    //cout << sonOFmask << endl;
	    cout << "Num OFlow Corners: " << sonOFnumCorners << endl;
	    cout << "Num OFlow Matches: " << sonOFnumMatches << endl;
	    cout << "Num OFlow Inliers: " << sonOFnumInliers << endl;
	    cout << "Sonar OFlow Model: " << sonOFmodel << endl;
	  }
	if(DISPLAY_IMGS)
	  {
	    putText(sonFeaturesImg, "Optical Flow", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    putText(sonMatchesImg, "Optical Flow", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    
	    imshow(son_features_wnd, sonFeaturesImg);	    
	    imshow(son_matches_wnd, sonMatchesImg);
	    if(STOP_BETWEEN_IMGS)
	      waitKey(0);
	    else
	      waitKey(10);
	  }

	if(WRITE_IMGS)
	  {
	    imwrite("OFsonarFeaturesIMG.jpg", sonFeaturesImg);
	    imwrite("OFsonarMatchesIMG.jpg", sonMatchesImg);
	  }


	//------------ SURF Feature Detection -----------------
	cvtColor(prev_sonarImgGray, sonFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	sonMatchesImg = prev_sonarImgGray.clone();//WRONG?? WON"T DISPLAY MATCHES?

	Mat sonSURFmodel, sonSURFmask;
	int sonSURFnumInliers = 0;
	int sonSURFnumMatches = 0;
	int sonSURFnumCorners = 0;
	std::vector<KeyPoint> sonSURFkeypoints_1, sonSURFkeypoints_2;
	std::vector< DMatch > sonSURFgood_matches;

	//Detect corners:
	sonSURFdetector.detect( prev_sonarImgGray, sonSURFkeypoints_1 );
	sonSURFdetector.detect( sonarImgGray, sonSURFkeypoints_2 );

	sonSURFnumCorners = sonSURFkeypoints_1.size();

	//SURF Calculate descriptors (feature vectors):                             
	Mat sonSURFdescriptors_1, sonSURFdescriptors_2;
	sonSURFextractor.compute( prev_sonarImgGray, sonSURFkeypoints_1, sonSURFdescriptors_1 );
	sonSURFextractor.compute( sonarImgGray, sonSURFkeypoints_2, sonSURFdescriptors_2 );

	//If any corners found, find matches:
	if((!sonSURFdescriptors_1.empty()) && (!sonSURFdescriptors_2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher 
	    std::vector< DMatch > sonSURFmatches;
	    sonSURFmatcher.match( sonSURFdescriptors_1, sonSURFdescriptors_2, sonSURFmatches );
	    if(USE_RADIUS_MATCH)
	      {
		double sonSURFmax_dist = 0; double sonSURFmin_dist = 100;
		//-- Quick calculation of max and min distances between keypoints
		for( int i1 = 0; i1 < sonSURFdescriptors_1.rows; i1++ )
		  { double dist = sonSURFmatches[i1].distance;
		    if( dist < sonSURFmin_dist ) sonSURFmin_dist = dist;
		    if( dist > sonSURFmax_dist ) sonSURFmax_dist = dist;
		  }
		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist, or a small arbitary value ( 0.02 ) in the event that min_dist is very small -- PS.- radiusMatch can also be used here.         
		for( int i1 = 0; i1 < sonSURFdescriptors_1.rows; i1++ )
		  { if( sonSURFmatches[i1].distance <= max(2*sonSURFmin_dist, 0.02))
		      { sonSURFgood_matches.push_back( sonSURFmatches[i1]); }
		  }   
	      }
	    else //Don't use radius match
	      {
		for( int i1 = 0; i1 < sonSURFdescriptors_1.rows; i1++ )
		  {
		    sonSURFgood_matches.push_back( sonSURFmatches[i1]);
		  }   
	      }

	    if(VERBOSE)
	      cout << "sonSURFgoodRadiusMatches=" << sonSURFgood_matches.size() << endl;

	    std::vector<Point2f> sonSURFgood_match_pts1;
	    std::vector<Point2f> sonSURFgood_match_pts2;
	    std::vector<Point2f> sonSURFgood_match_pts_diff;
	    for( int i1 = 0; i1 < sonSURFgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches     
		sonSURFgood_match_pts1.push_back( sonSURFkeypoints_1[ sonSURFgood_matches[i1].queryIdx ].pt );
		sonSURFgood_match_pts2.push_back( sonSURFkeypoints_2[ sonSURFgood_matches[i1].trainIdx ].pt );
		sonSURFgood_match_pts_diff.push_back(sonSURFkeypoints_2[sonSURFgood_matches[i1].trainIdx].pt - sonSURFkeypoints_1[sonSURFgood_matches[i1].queryIdx].pt);
	      }

	    sonSURFnumMatches = sonSURFgood_match_pts1.size();
	    
	    //Run RANSAC 
	    double sonSURF_RANSAC_reprojthresh = 1;
	    double sonSURF_RANSAC_param = 0.99;
	    int sonSURFok;
	    if(sonSURFgood_matches.size() >= 3) //If found enough points:
	      {
		if(SONAR_CARTESIAN)
		  sonSURFok = estimateRigidTransform2DNew(sonSURFgood_match_pts1, sonSURFgood_match_pts2, ROBUST_EST_METH,sonSURFmodel, sonSURFmask, sonSURF_RANSAC_reprojthresh, sonSURF_RANSAC_param, sonarSize);
		else //POLAR
		  sonSURFok = estimateTranslationNew(sonSURFgood_match_pts_diff, sonSURFgood_match_pts_diff, ROBUST_EST_METH, sonSURFmodel, sonSURFmask, sonSURF_RANSAC_reprojthresh, sonSURF_RANSAC_param);

		sonSURFnumInliers = sum(sonSURFmask)[0]; //Find number of inliers
		
		SURF_EstValid = 1; // Valid data flag

		if(DISPLAY_IMGS)
		  {
		    //Draw Features and Matches: 
		    drawMatches(prev_sonarImgGray, sonSURFkeypoints_1, sonarImgGray, sonSURFkeypoints_2, sonSURFgood_matches, sonMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		    for(int i1 = 0; i1 < sonSURFgood_match_pts2.size(); i1++)
		      {
			Scalar color;
			if(sonSURFmask.at<char>(0,i1))
			  color = Scalar(0,255,0); //Green
			else
			  color = Scalar(0,0,255); //Red
			circle(sonFeaturesImg, sonSURFgood_match_pts1[i1], 4, color);
		      }
		  }
	      }
	    else //Not enough matches
	      {
		cout << "NO SURF MODEL FOUND" << endl;
		sonSURFmodel = Mat::zeros(1,3,CV_64F);
		SURF_EstValid = 0; // Valid data flag
	      }
	  }
	else //No points found
	  {
	    cout << "NO SURF MODEL FOUND" << endl;
	    sonSURFmodel = Mat::zeros(1,3,CV_64F);
	    SURF_EstValid = 0; // Valid data flag
	  }

	if(VERBOSE)
	  {
	    cout << "Num SURF Corners: " << sonSURFnumCorners << endl;
	    cout << "Num SURF Matches: " << sonSURFnumMatches << endl;
	    cout << "Num SURF Inliers: " << sonSURFnumInliers << endl;
	    cout << "Sonar SURF Model" << sonSURFmodel << endl;
	  }
	if(DISPLAY_IMGS)
	  {
	    putText(sonFeaturesImg, "SURF", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    putText(sonMatchesImg, "SURF", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    imshow(son_features_wnd, sonFeaturesImg);
	    imshow(son_matches_wnd, sonMatchesImg);
	    if(STOP_BETWEEN_IMGS)
	      waitKey(0);
	    else
	      waitKey(10);
	  }
	if(WRITE_IMGS)
	  {
	    imwrite("SURFsonarFeaturesIMG.jpg", sonFeaturesImg);
	    imwrite("SURFsonarMatchesIMG.jpg", sonMatchesImg);
	  }


	//---------------Sonar SIFT Feature Detection ------------------//
	cvtColor(prev_sonarImgGray, sonFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features      
	sonMatchesImg = prev_sonarImgGray.clone();
        Mat sonSIFTmodel, sonSIFTmask;
	float sonSIFTnumInliers = 0;
	float sonSIFTnumMatches = 0;
	int sonSIFTnumCorners = 0;
	std::vector<KeyPoint> sonSIFTkeypoints_1, sonSIFTkeypoints_2;
	std::vector< DMatch > sonSIFTgood_matches;
	std::vector<Point2f> sonSIFTgood_match_pts1;
	std::vector<Point2f> sonSIFTgood_match_pts2;
	std::vector<Point2f> sonSIFTgood_match_pts_diff;

	//Detect Features:
        sonSIFTdetector->detect( prev_sonarImgGray, sonSIFTkeypoints_1 );
        sonSIFTdetector->detect( sonarImgGray, sonSIFTkeypoints_2 );

	sonSIFTnumCorners = sonSIFTkeypoints_1.size();

        //SURF Calculate descriptors (feature vectors):                             
        Mat sonSIFTdescriptors_1, sonSIFTdescriptors_2;
        sonSIFTextractor->compute( prev_sonarImgGray, sonSIFTkeypoints_1, sonSIFTdescriptors_1 );
        sonSIFTextractor->compute( sonarImgGray, sonSIFTkeypoints_2, sonSIFTdescriptors_2 );

	//If found features, try matching:
	if((!sonSIFTdescriptors_1.empty()) && (!sonSIFTdescriptors_2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher                  
	    std::vector< DMatch > sonSIFTmatches;
	    sonSIFTmatcher.match( sonSIFTdescriptors_1, sonSIFTdescriptors_2, sonSIFTmatches );
	    if(USE_RADIUS_MATCH)
	      {
		double sonSIFTmax_dist = 0; double sonSIFTmin_dist = 100;
		//-- Quick calculation of max and min distances between keypoints  
		for( int i1 = 0; i1 < sonSIFTdescriptors_1.rows; i1++ )
		  { double dist = sonSIFTmatches[i1].distance;
		    if( dist < sonSIFTmin_dist ) sonSIFTmin_dist = dist;
		    if( dist > sonSIFTmax_dist ) sonSIFTmax_dist = dist;
		  }
		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,  or a small arbitary value ( 0.02 ) in the event that min_dist is very small)   //-- PS.- radiusMatch can also be used here. 
		for( int i1 = 0; i1 < sonSIFTdescriptors_1.rows; i1++ )
		  { if( sonSIFTmatches[i1].distance <= max(2*sonSIFTmin_dist, 0.02))
		      sonSIFTgood_matches.push_back( sonSIFTmatches[i1]); 
		  }
	      }
	    else
	      {
		for( int i1 = 0; i1 < sonSIFTdescriptors_1.rows; i1++ )
		  {
		      sonSIFTgood_matches.push_back( sonSIFTmatches[i1]); 
		  }
	      }

	    for( int i1 = 0; i1 < sonSIFTgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches      
		sonSIFTgood_match_pts1.push_back( sonSIFTkeypoints_1[ sonSIFTgood_matches[i1].queryIdx ].pt );
		sonSIFTgood_match_pts2.push_back( sonSIFTkeypoints_2[ sonSIFTgood_matches[i1].trainIdx ].pt );
		sonSIFTgood_match_pts_diff.push_back( sonSIFTkeypoints_2[ sonSIFTgood_matches[i1].trainIdx].pt - sonSIFTkeypoints_1[ sonSIFTgood_matches[i1].queryIdx].pt);
	      }	  

	    sonSIFTnumMatches = sonSIFTgood_match_pts1.size();

	    //Run RANSAC                                            
	    double sonSIFT_RANSAC_reprojthresh = 1;
	    double sonSIFT_RANSAC_param = 0.99;
	    int sonSIFTok;
	    if(sonSIFTgood_matches.size() >= 3) //If found enough matches:
	      {
		if(SONAR_CARTESIAN)
		  sonSIFTok = estimateRigidTransform2DNew(sonSIFTgood_match_pts1, sonSIFTgood_match_pts2, ROBUST_EST_METH, sonSIFTmodel, sonSIFTmask, sonSIFT_RANSAC_reprojthresh, sonSIFT_RANSAC_param, sonarSize);
		else //POLAR
		  sonSIFTok = estimateTranslationNew(sonSIFTgood_match_pts_diff, sonSIFTgood_match_pts_diff, ROBUST_EST_METH, sonSIFTmodel, sonSIFTmask, sonSIFT_RANSAC_reprojthresh, sonSIFT_RANSAC_param);

		sonSIFTnumInliers = sum(sonSIFTmask)[0]; //Find number of inliers

		SIFT_EstValid = 1; // Valid data flag

		if(DISPLAY_IMGS)
		  {
		    //Draw Matches:        
		    drawMatches(prev_sonarImgGray, sonSIFTkeypoints_1, sonarImgGray, sonSIFTkeypoints_2, sonSIFTgood_matches, sonMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);                      
		    for(int i1 = 0; i1 < sonSIFTgood_match_pts1.size(); i1++)
		      {
			Scalar color;
			if(sonSIFTmask.at<char>(0,i1))
			  color = Scalar(0,255,0); //Green
			else
			  color = Scalar(0,0,255); //Red
			//Draw Features:
			circle(sonFeaturesImg, sonSIFTgood_match_pts1[i1], 4, color);
		      }
		  }
	      }
	    else //Didn't find enough matches
	      {
		cout << "NO SIFT MODEL FOUND" << endl;
		sonSIFTmodel = Mat::zeros(1,3,CV_64F);
		SIFT_EstValid = 0; // Valid data flag
	      }
	  }
	else //Didn't find any features 
	  {
	    cout << "NO SIFT MODEL FOUND" << endl;
	    sonSIFTmodel = Mat::zeros(1,3,CV_64F);
	    SIFT_EstValid = 0; // Valid data flag
	  }

	if(VERBOSE)
	  {
	    cout << "Num SIFT Corners: " << sonSIFTnumCorners << endl;
	    cout << "Num SIFT Matches: " << sonSIFTnumMatches << endl;
	    cout << "Num SIFT Inliers: " << sonSIFTnumInliers << endl;
	    cout << "sonar SIFT model" << sonSIFTmodel << endl;
	  }

	if(DISPLAY_IMGS)
	  {
	    putText(sonFeaturesImg, "SIFT", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    putText(sonMatchesImg, "SIFT", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    imshow(son_features_wnd, sonFeaturesImg);    
	    imshow(son_matches_wnd, sonMatchesImg);
	    if(STOP_BETWEEN_IMGS)
	      waitKey(0);
	    else
	      waitKey(10);
	  }
	if(WRITE_IMGS)
	  {
	    imwrite("SIFTsonarFeaturesIMG.jpg", sonFeaturesImg);
	    imwrite("SIFTsonarMatchesIMG.jpg", sonMatchesImg);
	  }
	//-------------------------- HARRIS/SIFT Detection ---------------//
	Mat sonHARRISmodel, sonHARRISmask;
	float sonHARRISnumMatches = 0;
	float sonHARRISnumInliers = 0;
	int sonHARRISnumCorners = 0;
	std::vector<cv::KeyPoint> sonHARRISkeypoints1, sonHARRISkeypoints2;
	cv::Mat sonHARRISdesc1, sonHARRISdesc2;
	std::vector< DMatch > sonHARRISmatches;
	std::vector< DMatch > sonHARRISgood_matches;
	std::vector<Point2f> sonHARRISgood_match_pts1;
	std::vector<Point2f> sonHARRISgood_match_pts2;
	std::vector<Point2f> sonHARRISgood_match_pts_diff;
	    
	cvtColor(prev_sonarImgGray, sonFeaturesImg, CV_GRAY2BGR); //Get copy of gray img to mark features
	sonMatchesImg = prev_sonarImgGray.clone();//DELETE??!!!!

	// detect keypoints
	sonHARRISdetector->detect(prev_sonarImgGray, sonHARRISkeypoints1);
	sonHARRISdetector->detect(sonarImgGray, sonHARRISkeypoints2);

	sonHARRISnumCorners = sonHARRISkeypoints1.size();

	// extract features
	sonHARRISdescriptor->compute(prev_sonarImgGray, sonHARRISkeypoints1, sonHARRISdesc1);
	sonHARRISdescriptor->compute(sonarImgGray, sonHARRISkeypoints2, sonHARRISdesc2);

	//If found features, try matching:
	if((!sonHARRISdesc1.empty()) && (!sonHARRISdesc2.empty()))
	  {
	    //SURF Matching descriptor vectors using FLANN matcher 
	    sonHARRISmatcher.match( sonHARRISdesc1, sonHARRISdesc2, sonHARRISmatches );
	    if(USE_RADIUS_MATCH)
	      {
		double sonHARRISmax_dist = 0; double sonHARRISmin_dist = 100;
		//-- Quick calculation of max and min distances between keypoints
		for( int i1 = 0; i1 < sonHARRISdesc1.rows; i1++ )
		  { double dist = sonHARRISmatches[i1].distance;
		    if( dist < sonHARRISmin_dist ) sonHARRISmin_dist = dist;
		    if( dist > sonHARRISmax_dist ) sonHARRISmax_dist = dist;
		  }
		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,-- or a small arbitary value ( 0.02 ) in the event that min_dist is very small)   //-- PS.- radiusMatch can also be used here.                  
		
		for( int i1 = 0; i1 < sonHARRISdesc1.rows; i1++ )
		  {
		    if(sonHARRISmatches[i1].distance <= max(2*sonHARRISmin_dist, 0.02))
		      sonHARRISgood_matches.push_back( sonHARRISmatches[i1]);
		  }
	      }
	    else //Don't use Radius Match
	      {
		for( int i1 = 0; i1 < sonHARRISdesc1.rows; i1++ )
		  {
		    sonHARRISgood_matches.push_back( sonHARRISmatches[i1]);
		  }
	      }

	    for( int i1 = 0; i1 < sonHARRISgood_matches.size(); i1++ )
	      {
		//-- Get the keypoints from the good matches    
		sonHARRISgood_match_pts1.push_back( sonHARRISkeypoints1[ sonHARRISgood_matches[i1].queryIdx ].pt );
		sonHARRISgood_match_pts2.push_back( sonHARRISkeypoints2[ sonHARRISgood_matches[i1].trainIdx ].pt );
		sonHARRISgood_match_pts_diff.push_back(sonHARRISkeypoints2[sonHARRISgood_matches[i1].trainIdx].pt - sonHARRISkeypoints1[sonHARRISgood_matches[i1].queryIdx].pt);
	      }

	    sonHARRISnumMatches = sonHARRISgood_match_pts1.size();

      	    //Run RANSAC                                                            
	    double sonHARRIS_RANSAC_reprojthresh = 1;
	    double sonHARRIS_RANSAC_param = 0.99;
	    int sonHARRISok;
	    if(sonHARRISgood_matches.size() >= 3) //If enough matches:
	      {
		if(SONAR_CARTESIAN)
		  sonHARRISok = estimateRigidTransform2DNew(sonHARRISgood_match_pts1, sonHARRISgood_match_pts2, ROBUST_EST_METH, sonHARRISmodel, sonHARRISmask, sonHARRIS_RANSAC_reprojthresh, sonHARRIS_RANSAC_param, sonarSize);
		else //POLAR
		  sonHARRISok = estimateTranslationNew(sonHARRISgood_match_pts_diff, sonHARRISgood_match_pts_diff, ROBUST_EST_METH, sonHARRISmodel, sonHARRISmask, sonHARRIS_RANSAC_reprojthresh, sonHARRIS_RANSAC_param);

		sonHARRISnumInliers = sum(sonHARRISmask)[0]; //Find number of inliers

		HARRIS_EstValid = 1; // Valid data flag

		if(DISPLAY_IMGS)
		  {
		    //Draw Matches:
		    drawMatches(prev_sonarImgGray, sonHARRISkeypoints1, sonarImgGray, sonHARRISkeypoints2, sonHARRISgood_matches, sonMatchesImg,Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		    for(int i1 = 0; i1 < sonHARRISgood_match_pts1.size(); i1++)
		      {
			Scalar color;
			if(sonHARRISmask.at<char>(0,i1))
			  color = Scalar(0,255,0); //Green
			else
			  color = Scalar(0,0,255); //Red
			//Draw Features:
			circle(sonFeaturesImg, sonHARRISgood_match_pts1[i1], 4, color);
		      }
		  }
	      }
	    else //Not enough matches
	      {
		cout << "NO HARRIS MODEL FOUND" << endl;
		sonHARRISmodel = Mat::zeros(1,3,CV_64F);
		HARRIS_EstValid = 0; // Valid data flag
	      }
	  }
	else //No corners found 
	  {
	    cout << "NO HARRIS MODEL FOUND" << endl;
	    sonHARRISmodel = Mat::zeros(1,3,CV_64F);
	    HARRIS_EstValid = 0; // Valid data flag
	  }

	if(VERBOSE)
	  {
	    cout << "Num HARRIS Corners: " << sonHARRISnumCorners << endl;
	    cout << "Num HARRIS Matches: " << sonHARRISnumMatches << endl;
	    cout << "Num HARRIS Inliers: " << sonHARRISnumInliers << endl;
	    cout << "Sonar HARRIS Model: " << sonHARRISmodel << endl;
	  }
	if(DISPLAY_IMGS)
	  {
	    putText(sonFeaturesImg, "HARRIS", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));
	    putText(sonMatchesImg, "HARRIS", Point(10,25),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255));

	//SONAR GLIB ERROR HERE::: But other places too. Seems to be a problem with the python?? libraries or where opencv is installed. check it out by googling exact error message.

	    imshow(son_features_wnd, sonFeaturesImg);
	    imshow(son_matches_wnd, sonMatchesImg);
	    if(STOP_BETWEEN_IMGS)
	      waitKey(0);
	    else
	      waitKey(10);
	  }
	//END SONAR GLIB ERROR
	if(WRITE_IMGS)
	  {
	    imwrite("HARRISsonarFeaturesIMG.jpg", sonFeaturesImg);
	    imwrite("HARRISsonarMatchesIMG.jpg", sonMatchesImg);
	  }

	//----------Convert angle from radians to degrees---------//
	//--------------------------------------------------------//
	sonOFmodel.at<double>(0,2) = sonOFmodel.at<double>(0,2)*180/PI;	
	sonSURFmodel.at<double>(0,2) = sonSURFmodel.at<double>(0,2)*180/PI;	
	sonSIFTmodel.at<double>(0,2) = sonSIFTmodel.at<double>(0,2)*180/PI;	
	sonHARRISmodel.at<double>(0,2) = sonHARRISmodel.at<double>(0,2)*180/PI;    

	//--------------------------------------------------------//
	//---------------Eliminate Large Jumps--------------------//
	if(THRESHOLD_JUMPS)
	  {
	    if((abs(sonOFmodel.at<double>(0,0)) > MAX_X)||
	       (abs(sonOFmodel.at<double>(0,1)) > MAX_Y)||
	       (abs(sonOFmodel.at<double>(0,2)) > MAX_PSI))
	      {
		cout << "OF Estimate out of range. Zeroed." << endl;
		cout << sonOFmodel << endl;
		sonOFmodel = Mat::zeros(1,3,CV_64F);
		OF_EstValid = 0; // Valid data flag
	      }
	    if((abs(sonSURFmodel.at<double>(0,0)) > MAX_X)||
	       (abs(sonSURFmodel.at<double>(0,1)) > MAX_Y)||
	       (abs(sonSURFmodel.at<double>(0,2)) > MAX_PSI))
	      {
		cout << "SURF Estimate out of range. Zeroed." << endl;
		cout << sonSURFmodel << endl;
		sonSURFmodel = Mat::zeros(1,3,CV_64F);
		SURF_EstValid = 0; // Valid data flag
	      }
	    if((abs(sonSIFTmodel.at<double>(0,0)) > MAX_X)||
	       (abs(sonSIFTmodel.at<double>(0,1)) > MAX_Y)||
	       (abs(sonSIFTmodel.at<double>(0,2)) > MAX_PSI))
	      {
		cout << "SIFT Estimate out of range. Zeroed." << endl;
		cout << sonSIFTmodel << endl;
		sonSIFTmodel = Mat::zeros(1,3,CV_64F);
		SIFT_EstValid = 0; // Valid data flag
	      }
	    if((sonHARRISmodel.at<double>(0,0) > MAX_X)||
	       (sonHARRISmodel.at<double>(0,1) > MAX_Y)||
	       (sonHARRISmodel.at<double>(0,2) > MAX_PSI))
	      {
		cout << "HARRIS Estimate out of range. Zeroed." << endl;
		cout << sonHARRISmodel << endl;
		sonHARRISmodel = Mat::zeros(1,3,CV_64F);
		HARRIS_EstValid = 0; // Valid data flag
	      }
	  }

	//--------------------------------------------------------//
	//---------------Get Simple Summation --------------------//
	sonOFsum[0] = sonOFsum[0] + sonOFmodel.at<double>(0,0);
	sonOFsum[1] = sonOFsum[1] + sonOFmodel.at<double>(0,1);
	sonOFsum[2] = sonOFsum[2] + sonOFmodel.at<double>(0,2);

	sonSURFsum[0] = sonSURFsum[0] + sonSURFmodel.at<double>(0,0);
	sonSURFsum[1] = sonSURFsum[1] + sonSURFmodel.at<double>(0,1);
	sonSURFsum[2] = sonSURFsum[2] + sonSURFmodel.at<double>(0,2);

	sonSIFTsum[0] = sonSIFTsum[0] + sonSIFTmodel.at<double>(0,0);
	sonSIFTsum[1] = sonSIFTsum[1] + sonSIFTmodel.at<double>(0,1);
        sonSIFTsum[2] = sonSIFTsum[2] + sonSIFTmodel.at<double>(0,2);

        sonHARRISsum[0] = sonHARRISsum[0] + sonHARRISmodel.at<double>(0,0);
        sonHARRISsum[1] = sonHARRISsum[1] + sonHARRISmodel.at<double>(0,1);
        sonHARRISsum[2] = sonHARRISsum[2] + sonHARRISmodel.at<double>(0,2);

	//--------------------------------------------------------//
	// -------------- Check for small nums/zeros -------------//
	int i2;
	for(i2=0; i2<3; i2++)
	  {
	    if(abs(sonOFmodel.at<double>(0,i2)) < ZERO)
	      sonOFmodel.at<double>(0,i2) = 0;
	    if(abs(sonSURFmodel.at<double>(0,i2)) < ZERO)
	      sonSURFmodel.at<double>(0,i2) = 0;
	    if(abs(sonSIFTmodel.at<double>(0,i2)) < ZERO)
	      sonSIFTmodel.at<double>(0,i2) = 0;
	    if(abs(sonHARRISmodel.at<double>(0,i2)) < ZERO)
	      sonHARRISmodel.at<double>(0,i2) = 0;

	    if(abs(sonOFsum[i2]) < ZERO)
	      sonOFsum[i2] = 0;
	    if(abs(sonSURFsum[i2]) < ZERO)
	      sonSURFsum[i2] = 0;
	    if(abs(sonSIFTsum[i2]) < ZERO)
	      sonSIFTsum[i2] = 0;
	    if(abs(sonHARRISsum[i2]) < ZERO)
	      sonHARRISsum[i2] = 0;
	  }
	
	//--------------------------------------------------------//
	//---------------Get Global x,y Sums ---------------------//
	//Note: Have to flip x,y here for vehicle coords
	/*	cout << "psi,sin,cos=" << sonOFsumGlobal[2] << "," << sin(sonOFsumGlobal[2]*PI/180) << "," << cos(sonOFsumGlobal[2]*PI/180) << endl;
	cout << "x,y=" << sonOFmodel.at<double>(0,1) << "," << sonOFmodel.at<double>(0,0) << endl;
	cout << "xcos,ysin=" << sonOFmodel.at<double>(0,1)*cos(sonOFsumGlobal[2]*PI/180) << ", " << sonOFmodel.at<double>(0,0)*sin(sonOFsumGlobal[2]*PI/180) << endl;
	cout << "xsin,ycos=" << ", " << sonOFmodel.at<double>(0,1)*sin(sonOFsumGlobal[2]*PI/180) << ", " << sonOFmodel.at<double>(0,0)*cos(sonOFsumGlobal[2]*PI/180) << endl;*/

	sonOFsumGlobal[2] = sonOFsumGlobal[2] - sonOFmodel.at<double>(0,2);
	sonOFsumGlobal[0] = sonOFsumGlobal[0] + sonOFmodel.at<double>(0,1)*cos(sonOFsumGlobal[2]*PI/180) + sonOFmodel.at<double>(0,0)*sin(sonOFsumGlobal[2]*PI/180);
	sonOFsumGlobal[1] = sonOFsumGlobal[1] + sonOFmodel.at<double>(0,1)*sin(sonOFsumGlobal[2]*PI/180) - sonOFmodel.at<double>(0,0)*cos(sonOFsumGlobal[2]*PI/180);

	//	cout << "sonOFsumGlobal[0,1,2]=" << sonOFsumGlobal[0] << "," << sonOFsumGlobal[1] << "," << sonOFsumGlobal[2] << endl;

	sonSURFsumGlobal[2] = sonSURFsumGlobal[2] - sonSURFmodel.at<double>(0,2);
	sonSURFsumGlobal[0] = sonSURFsumGlobal[0] + sonSURFmodel.at<double>(0,1)*cos(sonSURFsumGlobal[2]*PI/180) + sonSURFmodel.at<double>(0,0)*sin(sonSURFsumGlobal[2]*PI/180);
	sonSURFsumGlobal[1] = sonSURFsumGlobal[1] + sonSURFmodel.at<double>(0,1)*sin(sonSURFsumGlobal[2]*PI/180) - sonSURFmodel.at<double>(0,0)*cos(sonSURFsumGlobal[2]*PI/180);

	sonSIFTsumGlobal[2] = sonSIFTsumGlobal[2] - sonSIFTmodel.at<double>(0,2);
	sonSIFTsumGlobal[0] = sonSIFTsumGlobal[0] + sonSIFTmodel.at<double>(0,1)*cos(sonSIFTsumGlobal[2]*PI/180) + sonSIFTmodel.at<double>(0,0)*sin(sonSIFTsumGlobal[2]*PI/180);
	sonSIFTsumGlobal[1] = sonSIFTsumGlobal[1] + sonSIFTmodel.at<double>(0,1)*sin(sonSIFTsumGlobal[2]*PI/180) - sonSIFTmodel.at<double>(0,0)*cos(sonSIFTsumGlobal[2]*PI/180);

	sonHARRISsumGlobal[2] = sonHARRISsumGlobal[2] - sonHARRISmodel.at<double>(0,2);
	sonHARRISsumGlobal[0] = sonHARRISsumGlobal[0] + sonHARRISmodel.at<double>(0,1)*cos(sonHARRISsumGlobal[2]*PI/180) + sonHARRISmodel.at<double>(0,0)*sin(sonHARRISsumGlobal[2]*PI/180);
	sonHARRISsumGlobal[1] = sonHARRISsumGlobal[1] + sonHARRISmodel.at<double>(0,1)*sin(sonHARRISsumGlobal[2]*PI/180) - sonHARRISmodel.at<double>(0,0)*cos(sonHARRISsumGlobal[2]*PI/180);


	//--------------------------------------------------------//
	// -------------- Check for small nums/zeros -------------//
	for(i2=0; i2<3; i2++)
	  {
	    if(abs(sonOFsumGlobal[i2]) < ZERO)
	      sonOFsumGlobal[i2] = 0;
	    if(abs(sonSURFsumGlobal[i2]) < ZERO)
	      sonSURFsumGlobal[i2] = 0;
	    if(abs(sonSIFTsumGlobal[i2]) < ZERO)
	      sonSIFTsumGlobal[i2] = 0;
	    if(abs(sonHARRISsumGlobal[i2]) < ZERO)
	      sonHARRISsumGlobal[i2] = 0;
	  }


	//--------------------------------------------------------//
	//----------------Print Results to File ------------------//
	if((prev_i != i) || PROCESS_00_FRAME)
	  {
	    outfileSonar << fixed;
	    outfileSonar << setprecision(20);
	    outfileSonar << ping_time_sec << ";" << prev_i << ";" << i << ";";
	    outfileSonar << setprecision(3);
	    
	    //Print out raw data from shift estimation
	    if(PRINT_RAW_SHIFTS)
	      {
		outfileSonar << sonOFmodel.at<double>(0,0) << ";" << sonOFmodel.at<double>(0,1) << ";" << sonOFmodel.at<double>(0,2) << ";";
		outfileSonar << sonSURFmodel.at<double>(0,0) << ";" << sonSURFmodel.at<double>(0,1) << ";" << sonSURFmodel.at<double>(0,2) << ";";
		outfileSonar << sonSIFTmodel.at<double>(0,0) << ";" << sonSIFTmodel.at<double>(0,1) << ";" << sonSIFTmodel.at<double>(0,2) << ";";
		outfileSonar << sonHARRISmodel.at<double>(0,0) << ";" << sonHARRISmodel.at<double>(0,1) << ";" << sonHARRISmodel.at<double>(0,2) << ";";
	      }
    //Now print the vehicle coordinate shift estimates. This also goes to GTSAM out
	    //Note: Have to flip x,y here for vehicle coords. Also negate y,yaw
	    outfileSonar << sonOFmodel.at<double>(0,1) << ";" << -sonOFmodel.at<double>(0,0) << ";" << -sonOFmodel.at<double>(0,2) << ";";
	    outfileSonar << sonOFsum[1] << ";" << -sonOFsum[0] << ";" << -sonOFsum[2] << ";";
	    outfileSonar << sonOFsumGlobal[0] << ";" << sonOFsumGlobal[1] << ";" << sonOFsumGlobal[2] << ";";

	    outfileSonar << sonSURFmodel.at<double>(0,1) << ";" << -sonSURFmodel.at<double>(0,0) << ";" << -sonSURFmodel.at<double>(0,2) << ";";
	    outfileSonar << sonSURFsum[1] << ";" << -sonSURFsum[0] << ";" << -sonSURFsum[2] << ";";
	    outfileSonar << sonSURFsumGlobal[0] << ";" << sonSURFsumGlobal[1] << ";" << sonSURFsumGlobal[2] << ";";

	    outfileSonar << sonSIFTmodel.at<double>(0,1) << ";" << -sonSIFTmodel.at<double>(0,0) << ";" << -sonSIFTmodel.at<double>(0,2) << ";";
	    outfileSonar << sonSIFTsum[1] << ";" << -sonSIFTsum[0] << ";" << -sonSIFTsum[2] << ";";
	    outfileSonar << sonSIFTsumGlobal[0] << ";" << sonSIFTsumGlobal[1] << ";" << sonSIFTsumGlobal[2] << ";";

	    outfileSonar << sonHARRISmodel.at<double>(0,1) << ";" << -sonHARRISmodel.at<double>(0,0) << ";" << -sonHARRISmodel.at<double>(0,2) << ";";
	    outfileSonar << sonHARRISsum[1] << ";" << -sonHARRISsum[0] << ";" << -sonHARRISsum[2] << ";";
	    outfileSonar << sonHARRISsumGlobal[0] << ";" << sonHARRISsumGlobal[1] << ";" << sonHARRISsumGlobal[2] << ";";

	    outfileSonar << sonOFnumCorners << ";" << sonOFnumMatches << ";" << sonOFnumInliers << ";";
	    outfileSonar << sonSURFnumCorners << ";" << sonSURFnumMatches << ";" << sonSURFnumInliers << ";";
	    outfileSonar << sonSIFTnumCorners << ";" << sonSIFTnumMatches << ";" << sonSIFTnumInliers << ";";
	    outfileSonar << sonHARRISnumCorners << ";" << sonHARRISnumMatches << ";" << sonHARRISnumInliers << ";";

	    outfileSonar << OF_EstValid << ";"  << SURF_EstValid << ";"  << SIFT_EstValid << ";"  << HARRIS_EstValid << ";";
	    outfileSonar << endl;
	  }

	//--------------------------------------------------------//
	//----------------Print Results to GTSAM File ------------//
	//Note: Have to flip x,y here for vehicle coords. Also negate y,yaw
	outfileSonarGTSAM << fixed;
	outfileSonarGTSAM << setprecision(20);
	outfileSonarGTSAM << prev_ping_time_sec << ";" << ping_time_sec << ";";
	outfileSonarGTSAM << setprecision(3);
	outfileSonarGTSAM << sonOFmodel.at<double>(0,1) << ";" << -sonOFmodel.at<double>(0,0) << ";" << -sonOFmodel.at<double>(0,2) << ";" << sonOFnumCorners << ";" << sonOFnumMatches << ";" << sonOFnumInliers << ";";
	outfileSonarGTSAM << OF_EstValid << ";";//  << SURF_EstValid << ";"  << SIFT_EstValid << ";"  << HARRIS_EstValid << ";";
	outfileSonarGTSAM << endl;

	//----------------------------------------------------------//

	//Save new prev image
	prev_sonarImgGray = sonarImgGray.clone();
	prev_i = i;
	prev_ping_time_sec = ping_time_sec;

	//--------------------------------------------------------//
	//-------------------------------------------------------

	//Display Images:
	if(DISPLAY_IMGS)
	  {
	    if(WRITE_FRAME_NUM)
	      {
		//Put Frame Number on image
		char text[255];
		sprintf(text, "%d", i);
		cv::putText(colorImg, text, cvPoint(0,25), cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar::all(255));
	      }
	    imshow(color_wnd, colorImg);
	    //imshow(son_features_wnd, sonFeaturesImg);
	    //imshow(vid_features_wnd, vidFeaturesImg);
	  }
	if(VERBOSE)
	  cout << "Valid data (OF, SURF, SIFT, HARRIS)=" << OF_EstValid << "," << SURF_EstValid << ","<< SIFT_EstValid << ","<< HARRIS_EstValid << endl;

	if(WRITE_VIDEO)
	  {
	    //Write Video-----------------------------------------------
	    Mat outVideoImg = Mat::zeros(colorImg.rows*2,colorImg.cols*2,CV_8UC3);
	    Mat tmp1, tmp2, tmp3, tmp4, tmp5;

	    //vector<Mat> splitPlanes;
	    //split(colorImg,splitPlanes);
	    //colorImg.type = CV_8UC4
	    //cout << splitPlanes[3] << endl;
	    Mat tmp7; 
	    cvtColor(colorImg, tmp7, COLOR_BGRA2BGR);
	    //	    for(;;);
	    //	    resize(colorImg, tmp1, Size(width,height));
	    resize(tmp7, tmp1, Size(width,height));
	    Mat mapRoi1(outVideoImg, Rect(0, 0, colorImg.cols, colorImg.rows));
	    tmp1.copyTo(mapRoi1);
	    
	    cvtColor(sonarImgGray,tmp5,COLOR_GRAY2BGR);
	    resize(tmp5, tmp2, Size(width,height));
	    Mat mapRoi2(outVideoImg, Rect(width, 0, width, height));
	    tmp2.copyTo(mapRoi2);
	    
	    resize(sonMatchesImgOF, tmp3, Size(2*width,height));
	    Mat mapRoi3(outVideoImg, Rect(0, height, 2*width, height));
	    tmp3.copyTo(mapRoi3);
	    
	    outputVideo << outVideoImg;
	  }

	//-----------------------------------------------
	//Set valid data flags back to -1
	OF_EstValid = -1;
	SURF_EstValid = -1;
	SIFT_EstValid = -1;
	HARRIS_EstValid = -1;
	
	BVTPing_Destroy(ping);
	  
	//------------------------------------
	//Check for key press:
	if(pause)
	{
	  while(1){
		keypushed = waitKey(10);
		if(keypushed==27)
			break;
		else if(keypushed=='p')
		{
			pause=false;
			break;
		}
		else if(keypushed=='f')
		{
		  //Go Forward 1 frame - do nothing because will be auto incremented
		  break;
		}
		else if(keypushed=='b')
		{
			i=i-2;	//Go back 2 frames (really going back one frame)
			break;
		}
		else if(tbar_update)
		  {
		    tbar_update = false;
		    i=i-1;
		    break;
		  }
	  }
	  if(keypushed==27)
	    break;
	}
	else
	{
		keypushed = waitKey(10);
		if(keypushed==27)
			break;
		else if(keypushed=='p')
		{
			pause=true;
			i = i-1; //decrement because will be auto incremented back	
		}	
	}
	//------------------------------------
	sonarSizeLast = Size(sonarSize.width,sonarSize.height);
     }

     // Clean up
     destroyWindow(color_wnd);
     destroyWindow(son_features_wnd);	
     destroyWindow(son_matches_wnd);

     if(!VIDEO_SONAR_FILE)
       {
	 BVTColorMapper_Destroy(mapper);
	 BVTSonar_Destroy(son);
       }

     outfileSonar.close();
     outfileSonarGTSAM.close();

     return 0;
}
