//============================================================================
// Name        : ChromaticAberration_Correction.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Correct chromatic aberration
//============================================================================

#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>      //for imshow
#include <vector>
#include <iostream>
#include <iomanip>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/photo.hpp>


using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

const double akaze_thresh = 0.5e-3; // AKAZE detection threshold
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio

const bool print_debug_images=false;
const bool align_optical_circles=true;
const bool use_CLAHE=false;

namespace example {
class Tracker
{
public:
    Tracker(Ptr<AKAZE> _detector, Ptr<DescriptorMatcher> _matcher) :
        detector(_detector),
        matcher(_matcher)
    {}

    void setReferenceChannel(const Mat frame);
    void process(const Mat im_channel_original, Mat &im_channel_corrected, Mat &homography, Mat &im_matches );
    Mat segmentationOpticalCircle(Mat resultRGB, string filename, int channel);
    Mat getConvexHull(Mat OpticalCircleMask);
    Mat FillChildBlobs(Mat inputBin);
    Mat getMinEnclosingCircle(Mat OpticalCircleMask);
    vector<String> getImagesPaths(string input_path);
    Ptr<Feature2D> getDetector() {
        return detector;
    }
protected:
    Ptr<AKAZE> detector;
    Ptr<DescriptorMatcher> matcher;
    Mat first_frame, first_desc;
    vector<KeyPoint> first_kp;
};

void Tracker::setReferenceChannel(const Mat frame)
{

    first_frame = frame.clone();
    detector->detectAndCompute(first_frame, Mat(), first_kp, first_desc);

}

void Tracker::process(const Mat im_channel_original, Mat &im_channel_corrected, Mat &homography, Mat &im_matches )
{
	//Initializations
    vector<KeyPoint> kp;
    Mat desc;

    // Detect AKAZE features and compute descriptors.
    detector->detectAndCompute(im_channel_original, Mat(), kp, desc);

    // Match features
//    vector< vector<DMatch> > matches;
//    vector<KeyPoint> matched1, matched2;
//    std::vector<Point2f> points1, points2;

//    matcher->knnMatch(first_desc, desc, matches, 2);
//    for(unsigned i = 0; i < matches.size(); i++) {
////        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
//            matched1.push_back(first_kp[matches[i][0].queryIdx]);
//            matched2.push_back(      kp[matches[i][0].trainIdx]);
//            points1.push_back( first_kp[matches[i][0].queryIdx].pt );
//            points2.push_back( kp[matches[i][0].trainIdx].pt );
////        }
//    }



    std::vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(desc,first_desc, matches, Mat());


//    int maximum_matches_number = 3000;
//    int current_matches_number = matches.size();
//    cout << "matches >> " << matches.size() << endl;
//
//    //Validate if number of matches is below of maximum matches threshold
//    if (current_matches_number>maximum_matches_number){
//    	float GOOD_MATCH_PERCENT = (float)maximum_matches_number/(float)current_matches_number;
//
//    	// Sort matches by score
//        std::sort(matches.begin(), matches.end());
//
//        // Remove not so good matches
//        const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
//        matches.erase(matches.begin()+numGoodMatches, matches.end());
//
//        cout << "matches filtered>> " << matches.size() << endl;
//    }


    const double kDistanceCoef = 70.0;
    const int kMaxMatchingSize = 7000;
    cout << "matches >> " << matches.size() << endl;

    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    //Filter by distance
    while (matches.front().distance * kDistanceCoef < matches.back().distance) {
        matches.pop_back();
    }

    //Filter by number of matches
    while (matches.size() > kMaxMatchingSize) {
        matches.pop_back();
    }

        cout << "matches filtered>> " << matches.size() << endl;


    // Draw top matches
      Mat imMatches;
      drawMatches(im_channel_original, kp, first_frame, first_kp, matches, im_matches);
      // Extract location of good matches
      std::vector<Point2f> points1, points2;

      for( size_t i = 0; i < matches.size(); i++ )
      {
        points1.push_back( kp[ matches[i].queryIdx ].pt );
        points2.push_back( first_kp[ matches[i].trainIdx ].pt );
      }

      // Find homography
//      homography = findHomography( points1, points2, RANSAC, ransac_thresh );
      homography = findHomography( points1, points2, USAC_MAGSAC);


      // Use homography to warp image
      warpPerspective(im_channel_original, im_channel_corrected, homography, first_frame.size());


}

Mat Tracker::segmentationOpticalCircle(Mat resultRGB, string filename, int channel) {

	//Mat resultBin;
	Mat OpticalCircleMask;
	int numberOfChannels=resultRGB.channels();

	//Convert RGB to Gray
	if (numberOfChannels>1){
//		cvtColor(resultRGB,resultGray,CV_BGR2GRAY,1);
		extractChannel(resultRGB,OpticalCircleMask,channel);
	}
	else {
		OpticalCircleMask = resultRGB;
	}

    // Window size for pre-processing operations (must be odd!)
	int heightc = OpticalCircleMask.rows;
	int widthc = OpticalCircleMask.cols;
    int window,windowFactor;
    windowFactor=80;
    if (widthc >= heightc) {
    	window =  widthc / windowFactor;
    }
    else {
    	window = heightc / windowFactor;
    }

    //Set window to odd value
    int remainder = (int)window % 2;
    if (remainder == 0) {
    	window = window + 1;
    }

	//Otsus Segmentation
	threshold(OpticalCircleMask, OpticalCircleMask, 128, 255, THRESH_BINARY_INV | THRESH_OTSU );

	//Delete inner structures
	Mat innerStructures_mask;
	OpticalCircleMask.copyTo(innerStructures_mask);

		//Delete black area
		rectangle(innerStructures_mask,Point(0,0),Point(widthc,0),Scalar(255),10);
		floodFill(innerStructures_mask,Point(0,0),Scalar(0),0,Scalar(), Scalar(), 8);
		rectangle(innerStructures_mask,Point(0,heightc-1),Point(widthc,heightc),Scalar(255),10);
		floodFill(innerStructures_mask,Point(0,heightc-1),Scalar(0),0,Scalar(), Scalar(), 8);

		//Delete structures
		OpticalCircleMask= OpticalCircleMask-innerStructures_mask;
		OpticalCircleMask=~OpticalCircleMask;
		innerStructures_mask.release();


		//Get minimum enclosed circle
		OpticalCircleMask= getMinEnclosingCircle(OpticalCircleMask);

//		Get Convex Hull mask
//		OpticalCircleMask = getConvexHull(OpticalCircleMask);

    return OpticalCircleMask;
}

Mat Tracker::getMinEnclosingCircle(Mat OpticalCircleMask){

	//Initializations
    vector<vector<Point> > contours;
    Mat OpticalCircleMask_aux;
    OpticalCircleMask.copyTo(OpticalCircleMask_aux);

    //Find contours
	findContours( OpticalCircleMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Point2f>centers( contours.size() );
	vector<float>radius( contours.size() );

	//Get minimum enclosing circle
	float max_radius_index=0, max_radius=0;

	for( size_t i = 0; i < contours.size(); i++ )
	{
		approxPolyDP( contours[i], contours_poly[i], 3, true );
		boundRect[i] = boundingRect( contours_poly[i] );
		minEnclosingCircle( contours_poly[i], centers[i], radius[i] );

		if( radius[i]>max_radius){
			max_radius= radius[i];
			max_radius_index=i;
		}
	}

	//Draw circle
	Scalar color = Scalar(255);
	circle( OpticalCircleMask_aux, centers[max_radius_index], (int)radius[max_radius_index]-4, color, 2 );

	//Fill child blobs
	OpticalCircleMask_aux= FillChildBlobs(OpticalCircleMask_aux);

	//Set black regions to 0 (for cases where the enclosing circle is bigger than the image width)
	if (max_radius>(OpticalCircleMask.cols/2)){
		for (int i=0;i<OpticalCircleMask.cols;i++){
			if(countNonZero(OpticalCircleMask.col(i))==0){
				OpticalCircleMask_aux.col(i).setTo(0);
			}
		}
	}

	return OpticalCircleMask_aux;
}

Mat Tracker::getConvexHull(Mat OpticalCircleMask) {

	// Convex Hull implementation
	  Mat src_copy = OpticalCircleMask.clone();

	  // contours vector
	  vector< vector<Point> > contours;
	  vector<Vec4i> hierarchy;

	  // find contours for the thresholded image
	  findContours(OpticalCircleMask, contours, hierarchy, RETR_TREE,
	      CHAIN_APPROX_SIMPLE, Point(0, 0));

	  // create convex hull vector
	  vector< vector<Point> > hull(contours.size());

	  // find convex hull for each contour
	  for(int i = 0; i < contours.size(); i++)
	    convexHull(Mat(contours[i]), hull[i], false);


	  // draw contours and convex hull on the empty black image
	  for(int i = 0; i < contours.size(); i++) {
	//		    Scalar color_contours = Scalar(0, 255, 0); // color for contours : blue
	    Scalar color = Scalar(255, 255, 255); // color for convex hull : white
	//		    // draw contours
	//		    drawContours(drawing, contours, i, color_contours, 2, 8, vector<Vec4i>(), 0, Point());
	    // draw convex hull
	    drawContours(OpticalCircleMask, hull, i, color, FILLED, 8, vector<Vec4i>(), 0, Point());
	  }

//		//Save images
//		string IMAGES_PATH_optical_circle_merged_convexHull =output_path+image_name+"_optical_circle_Z_merged_convexHull.jpg";
//		imwrite(IMAGES_PATH_optical_circle_merged_convexHull, OpticalCircleMask);

	return OpticalCircleMask;

}

Mat Tracker::FillChildBlobs(Mat inputBin) {

	//Create auxiliary Mats
	Mat resultBin;
	inputBin.copyTo(resultBin);

	int height = resultBin.rows;
	int width = resultBin.cols;

	//Invert binary image
	Mat resultBinInv;
	resultBin.copyTo(resultBinInv);
	resultBinInv = ~resultBinInv;

	//Delete blobs that touches the image borders
	floodFill(resultBinInv,Point(0,0),Scalar(0),0,Scalar(), Scalar(), 8);
	floodFill(resultBinInv,Point(width-1,height-1),Scalar(0),0,Scalar(), Scalar(), 8);

	// Add child blobs to binary image
	Mat aux;
	aux = resultBin + resultBinInv;
	aux.copyTo(resultBin);

	return resultBin;
}

vector<String> Tracker::getImagesPaths(string input_path){

    //Get paths of all images
    vector<String> vector_jpeg_path, vector_jpg_path, vector_images_path;
    String jpeg_path = input_path+"\\*.jpeg";
    String jpg_path = input_path+"\\*.jpg";

    glob(jpeg_path, vector_jpeg_path, false);
    glob(jpg_path, vector_jpg_path, false);

    vector_images_path.reserve( vector_jpeg_path.size() + vector_jpg_path.size() ); // preallocate memory
    vector_images_path.insert( vector_images_path.end(), vector_jpeg_path.begin(), vector_jpeg_path.end() );
    vector_images_path.insert( vector_images_path.end(), vector_jpg_path.begin(), vector_jpg_path.end() );

    return vector_images_path;
}

}

int main(int argc, char **argv)
{

	//Initializations
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->setThreshold(akaze_thresh);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    example::Tracker Akaze_ChromaticAberrations_Corrector(akaze, matcher);

    //Set input path
    const string input_path = "C:\\Users\\luis.rosado\\Documents\\TAMI\\Chromatic_Aberration_Correction\\cytology\\";
    //Set output path
    const string output_path = "C:\\Users\\luis.rosado\\Documents\\TAMI\\Chromatic_Aberration_Correction\\cytology\\results\\";

    //Get images Paths
    vector<String> vector_images_path = Akaze_ChromaticAberrations_Corrector.getImagesPaths(input_path);

    //Iterate for each image
    size_t count = vector_images_path.size(); //number of images files in target folder
    for (size_t i=0; i<count; i++){

        cout << (vector_images_path[i]) << endl;
//       vector_images_path[i] = "C:\\Users\\luis.rosado\\Documents\\TAMI\\Chromatic_Aberration_Correction\\SELECTED\\5b48a1f7-d2bd-450a-8108-62b9c7523e27_original.jpeg";

    	//Get configuration path
        String image_name;
        unsigned initPos = vector_images_path[i].find_last_of("/\\")+1;
    	unsigned finalPos = vector_images_path[i].find(".j");      // position of "live" in str
    	unsigned length = finalPos-initPos;
    	image_name=vector_images_path[i].substr(initPos,length);

        cout << (image_name) << endl;

		//Load image
		Mat im_original =  imread(vector_images_path[i]);

		if(print_debug_images){
	    	//Save image
			const string IMAGES_PATH_original = output_path+image_name+".jpg";
			imwrite(IMAGES_PATH_original,im_original);
		}

    //Split RGB channels
	vector<Mat> channels_original;
    split(im_original,channels_original);

    	if(print_debug_images){
    		//Save RGB channels images
    		const string IMAGES_PATHB = output_path+image_name+"__B.jpg";
    		const string IMAGES_PATHG = output_path+image_name+"__G.jpg";
    		const string IMAGES_PATHR = output_path+image_name+"__R.jpg";

    		imwrite(IMAGES_PATHB,channels_original[0]);
    		imwrite(IMAGES_PATHG,channels_original[1]);
    		imwrite(IMAGES_PATHR,channels_original[2]);
    	}

    //Get keypoints and descriptor of template channel (Green channel)
    Akaze_ChromaticAberrations_Corrector.setReferenceChannel(channels_original[1]);

    //Align Blue channel with template channel (Green channel)
    Mat channelB_corrected, homography_B, matches_B;
    Akaze_ChromaticAberrations_Corrector.process(channels_original[0], channelB_corrected, homography_B, matches_B);

		if(print_debug_images){
			//Save images
			const string output_path_B_corrected = output_path+image_name+"__B_correct.jpg";
			imwrite(output_path_B_corrected, channelB_corrected);
			const string output_path_B_matches = output_path+image_name+"__B_matches.jpg";
			imwrite(output_path_B_matches, matches_B);
		}

    //Align Red channel with template channel (Green channel)
    Mat channelR_corrected, homography_R, matches_R;
    Akaze_ChromaticAberrations_Corrector.process(channels_original[2], channelR_corrected, homography_R, matches_R);

	if(print_debug_images){
		//Save images
		const string output_path_R_corrected = output_path+image_name+"__R_correct.jpg";
		imwrite(output_path_R_corrected, channelR_corrected);
		const string output_path_R_matches = output_path+image_name+"__R_matches.jpg";
		imwrite(output_path_R_matches, matches_R);
	}

	//Create RGB image with corrected images
    vector<Mat> channels_corrected;
    channels_corrected.push_back(channelB_corrected);
    channels_corrected.push_back(channels_original[1]);
    channels_corrected.push_back(channelR_corrected);
    Mat im_corrected;
    merge(channels_corrected, im_corrected);

		if(print_debug_images){
			//Save images
			const string output_path_corrected1 = output_path+image_name+"___correct1.jpg";
			imwrite(output_path_corrected1, im_corrected);

		}

    //Align optical circles of each corrected channels
	if(align_optical_circles){

		string IMAGES_PATH_optical_circle_B =output_path+image_name+"_optical_circle_B.jpg";
		string IMAGES_PATH_optical_circle_G =output_path+image_name+"_optical_circle_G.jpg";
		string IMAGES_PATH_optical_circle_R =output_path+image_name+"_optical_circle_R.jpg";
		Mat optical_circle_B = Akaze_ChromaticAberrations_Corrector.segmentationOpticalCircle(im_corrected, IMAGES_PATH_optical_circle_B,0);
		Mat optical_circle_G = Akaze_ChromaticAberrations_Corrector.segmentationOpticalCircle(im_corrected, IMAGES_PATH_optical_circle_G,1);
		Mat optical_circle_R = Akaze_ChromaticAberrations_Corrector.segmentationOpticalCircle(im_corrected, IMAGES_PATH_optical_circle_R,2);

			if(print_debug_images){
				//Save images
				imwrite(IMAGES_PATH_optical_circle_B, optical_circle_B);
				imwrite(IMAGES_PATH_optical_circle_G, optical_circle_G);
				imwrite(IMAGES_PATH_optical_circle_R, optical_circle_R);
			}

		//Get minimum value of each optical circle mask
		Mat optical_circle_merged;
		min(optical_circle_B,optical_circle_G,optical_circle_merged);
		min(optical_circle_merged,optical_circle_R,optical_circle_merged);

			if(print_debug_images){
				//Save images
				string IMAGES_PATH_optical_circle_merged =output_path+image_name+"_optical_circle_Z_merged.jpg";
				imwrite(IMAGES_PATH_optical_circle_merged, optical_circle_merged);
			}

			//Get minimum value of each optical circle mask
			Mat optical_circle_maximum;
			max(optical_circle_B,optical_circle_G,optical_circle_maximum);
			max(optical_circle_maximum,optical_circle_R,optical_circle_maximum);

			double optical_circle_ratio = (double) countNonZero(optical_circle_merged)	/ (double) countNonZero(optical_circle_maximum);
			cout << "Optical circle ratio >> "<< optical_circle_ratio << endl;
			double optical_circle_threshold = 0.94;

		//Validated merge of optical circle
		if(optical_circle_ratio<optical_circle_threshold){
			cout << "Optical circle validation: NOT PASSED!!!!" << endl;
		}

		//Get image with minimum intensity for each pixel position for the corrected image
		Mat im_corrected_min;
		min(channels_corrected[0],channels_corrected[1],im_corrected_min);
		min(im_corrected_min,channels_corrected[2],im_corrected_min);

		im_corrected_min.copyTo(channels_corrected[0],optical_circle_merged==0);
		im_corrected_min.copyTo(channels_corrected[1],optical_circle_merged==0);
		im_corrected_min.copyTo(channels_corrected[2],optical_circle_merged==0);
		merge(channels_corrected, im_corrected);

				if(print_debug_images){
				//Save images
				const string output_path_corrected2 = output_path+image_name+"___correct2.jpg";
				imwrite(output_path_corrected2, im_corrected);
				}
	}


	// Apply the CLAHE algorithm to the B channel
	if(use_CLAHE){

		Mat im_clahe, im_clahe_binary;
	    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	    clahe->setClipLimit(1);
	    clahe->apply(channels_corrected[0], im_clahe);
	    threshold(im_clahe, im_clahe_binary, 205, 255, THRESH_BINARY);

	    double CLAHE_binary_ratio_threshold = 0.02;
	    double CLAHE_binary_ratio = (double)countNonZero(im_clahe_binary)/(double)(im_clahe.rows*im_clahe.cols);
	    cout << "CLAHE ratio >> " << CLAHE_binary_ratio << endl;

        Mat inpainted;
        if(CLAHE_binary_ratio<CLAHE_binary_ratio_threshold){
        	//Apply inpaint
            inpaint(channels_corrected[0], im_clahe_binary, inpainted, 6, INPAINT_TELEA);

            //Update channel
            channels_corrected[0]=inpainted;
        }else{

    		cout << "CLAHE ratio validation: NOT PASSED!!!!" << endl;

            channels_corrected[0].copyTo(inpainted);
        }

			if(print_debug_images){
				//Save images
				const string output_path_B_corrected_CLAHE = output_path+image_name+"__B_correct_CLAHE.jpg";
				imwrite(output_path_B_corrected_CLAHE, im_clahe);
				const string output_path_B_corrected_CLAHE2 = output_path+image_name+"__B_correct_CLAHE2.jpg";
				imwrite(output_path_B_corrected_CLAHE2, im_clahe_binary);
				const string output_path_B_corrected_CLAHE3 = output_path+image_name+"__B_correct_CLAHE3.jpg";
				imwrite(output_path_B_corrected_CLAHE3, inpainted);
			}

		//merge updated channels
		merge(channels_corrected, im_corrected);
	}


		if(print_debug_images){
			//Save images
			const string output_path_corrected3 = output_path+image_name+"___correct3.jpg";
			imwrite(output_path_corrected3, im_corrected);
		}
		else{
			//Save images
			const string output_path_corrected3 = output_path+image_name+"_corrected2.jpeg";
			imwrite(output_path_corrected3, im_corrected);
			const string output_path_original = output_path+image_name+".jpeg";
			imwrite(output_path_original, im_original);
		}

		cout << "Finished!" << endl;


//	//Super Resolution
//
//		// Region to crop
//		Rect roi;
//		roi.x = im_corrected.size().width/3;
//		roi.y = im_corrected.size().height/3.5;
//		roi.width = im_corrected.size().width/3;
//		roi.height = im_corrected.size().width/3;
//		im_corrected = im_corrected(roi);
//
//		//Save images
//		const string output_path_corrected2_cropped = output_path+image_name+"___correct2_cropped.jpg";
//		imwrite(output_path_corrected2_cropped, im_corrected);
//
//
//	//Create the module's object
//	DnnSuperResImpl sr;
//
//	//Read the desired model
//	string path = "LapSRN_x8.pb";
//	sr.readModel(path);
//	int scale = 8;
//
//	//Set the desired model and scale to get correct pre- and post-processing
//	sr.setModel("lapsrn", scale);
//
//	//Upscale
//	Mat im_superresolution;
//	sr.upsample(im_corrected, im_superresolution);
//
//		//Save images
//		const string output_path_corrected2_super = output_path+image_name+"___correct2_super.jpg";
//		imwrite(output_path_corrected2_super, im_superresolution);
//
//
//	// Image resized using OpenCV
//	Mat resized;
//	cv::resize(im_corrected, resized, cv::Size(), scale, scale);
//
//		//Save images
//		const string output_path_corrected2_resized = output_path+image_name+"___correct2_resized.jpg";
//		imwrite(output_path_corrected2_resized, resized);
//
//
//	// Image downsample using OpenCV
//	Mat downsampled;
//	double scale_downsample = 1/(double)scale;
//	cv::resize(im_superresolution, downsampled, cv::Size(), scale_downsample, scale_downsample);
//
//		//Save images
//		const string output_path_corrected4_downsampled = output_path+image_name+"___correct3_super_downsampled.jpg";
//		imwrite(output_path_corrected4_downsampled, downsampled);


    }

    return 0;
}
