//#include "precomp.hpp"
#include "RANSAC.hpp"
#include "_modelest.h"
//#include <algorithm>
//#include <iterator>
//#include <limits>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


/*namespace cv
{

class TranslationEstimator : public CvModelEstimator2
{
public:
    TranslationEstimator() : CvModelEstimator2(1, cvSize(2, 1), 1) {}
  virtual int runKernel( const CvMat* m1, const CvMat* m2, CvMat* model );
  //virtual bool runRANSAC(const CvMat* m1, const CvMat* m2, CvMat* model,CvMat* mask0, double reprojThreshold, double confidence, int maxIters );
  virtual bool runMAPSAC(const CvMat* m1, const CvMat* m2, CvMat* model,CvMat* mask0, double reprojThreshold, double confidence, int maxIters );
protected:
    virtual void computeReprojError( const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error );

};

}*/ //REMOVED THIS FOR VERSION 3.0.0 ^^^^^^^^^^^^^^^^^^^^^^^^





/*
bool TranslationEstimator::runRANSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
				   CvMat* mask0, double reprojThreshold,
				   double confidence, int maxIters )
{

  bool result = false;
  cv::Ptr<CvMat> mask = cvCloneMat(mask0);
  cv::Ptr<CvMat> models, err, tmask;
  cv::Ptr<CvMat> ms1, ms2;

  int iter, niters = maxIters;
  int count = m1->rows*m1->cols, maxGoodCount = 0;

  CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

  if( count < modelPoints )
    return false;

  models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
  err = cvCreateMat( 1, count, CV_32FC1 );
  tmask = cvCreateMat( 1, count, CV_8UC1 );

  if( count > modelPoints )
    {
      ms1 = cvCreateMat( 1, modelPoints, m1->type );
      ms2 = cvCreateMat( 1, modelPoints, m2->type );
    }
  else
    {
      niters = 1;
      ms1 = cvCloneMat(m1);
      ms2 = cvCloneMat(m2);
    }

  for( iter = 0; iter < niters; iter++ )
    {
      int i, goodCount, nmodels;
      if( count > modelPoints )
        {
	  bool found = getSubset( m1, m2, ms1, ms2, 300 );
	  if( !found )
            {
	      if( iter == 0 )
		return false;
	      break;
            }
        }

      nmodels = runKernel( ms1, ms2, models );
      if( nmodels <= 0 )
	continue;
      for( i = 0; i < nmodels; i++ )
        {
	  CvMat model_i;
	  cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );

	  goodCount = findInliers( m1, m2, &model_i, err, tmask, reprojThreshold );
	  cout << "AMS GOOD COUNT " << goodCount << endl;
	  if( goodCount > MAX(maxGoodCount, modelPoints-1) )
            {
	      std::swap(tmask, mask);
	      cv::Ptr<CvMat> temp1, temp2; //AMS
	      temp1 = &model_i;
	      cout << "AMS ABOUT TO COPY " << endl;
	      cout << "ROWS: " << temp1->rows << " AND  " << model->rows << endl;
	      cout << "COLS: " << temp1->cols << " AND " << model->cols << endl;
	      cvCopy( &model_i, model );
	      
	      cout << "AMS DONE COPY " << endl;
	      maxGoodCount = goodCount;
	      niters = cvRANSACUpdateNumIters( confidence,
					       (double)(count - goodCount)/count, modelPoints, niters );
            }
        }
    }

  if( maxGoodCount > 0 )
    {
      if( mask != mask0 )
	cvCopy( mask, mask0 );
      result = true;
    }

  return result;
}
*/



 /*int cv::TranslationEstimator::runKernel( const CvMat* m1, const CvMat* m2, CvMat* model )
{
  //cout << "DEBUG:ENTER RUNKERNEL" << endl;
    const Point2d* from = reinterpret_cast<const Point2d*>(m1->data.ptr);

    //cout << "ROWS = " << model->rows << endl;
    //cout << "COLS = " << model->cols << endl;
    Mat modelMat = Mat::zeros(1,2,CV_64F);
 
    modelMat.at<double>(0,0) = from[0].x;
    modelMat.at<double>(0,1) = from[0].y;

    CvMat cvmat_model = modelMat;

    //    model = &cvmat_model;
    CvMat cvX;
    cvReshape(model, &cvX, 1, 2);
    double* cvXptr = reinterpret_cast<double*>(model->data.ptr);
    Point2d* modelptr = reinterpret_cast<Point2d*>(model->data.ptr);
    cvXptr[0] = from[0].x;
    cvXptr[1] = from[0].y;

    //cout << "RunKernel Model = " << modelptr[0] << endl;
    //cout << "ROWS = " << model->rows << endl;
    //cout << "COLS = " << model->cols << endl;
    return 1;
}

void cv::TranslationEstimator::computeReprojError( const CvMat* m1, const CvMat* m2, const CvMat* model, CvMat* error )
{
  //cout << "DEBUG:ENTER COMPUTEREPROJERROR" << endl;
    int count = m1->rows * m1->cols;
    const Point2d* from = reinterpret_cast<const Point2d*>(m1->data.ptr);
    const Point2d* to   = reinterpret_cast<const Point2d*>(m2->data.ptr);
    const double* F = model->data.db;
    float* err = error->data.fl;

    for(int i = 0; i < count; i++ )
    {
        const Point2d& f = from[i];
        //const Point2d& t = to[i];

        //double a = F[0]*f.x + F[1]*f.y + F[ 2]*f.z + F[ 3] - t.x;
        //double b = F[4]*f.x + F[5]*f.y + F[ 6]*f.z + F[ 7] - t.y;
        //double c = F[8]*f.x + F[9]*f.y + F[10]*f.z + F[11] - t.z;
	
	double xshift = f.x - F[0];
	double yshift = f.y - F[1];

        err[i] = (float)sqrt(xshift*xshift + yshift*yshift);

	//cout << "Error " << err[i] << endl;
    }
    //cout << "DEBUG LEAVING REPROJ ERR" << endl;
}

bool TranslationEstimator::runMAPSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
				   CvMat* mask, double reprojThreshold, double confidence, int maxIters )
{
  //const double outlierRatio = 0.45;
  bool result = false;
  CvMat* models;
  CvMat* ms1;
  CvMat* ms2;
  CvMat* err;

  int iter, niters = maxIters;
  int count = m1->rows*m1->cols;
  double minSSE = DBL_MAX, sigma;

  CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

  if( count < modelPoints )
    return false;

  int tmp_height = modelSize.height*maxBasicSolutions;
  int tmp_width = modelSize.width;

  models = cvCreateMat( tmp_height, tmp_width, CV_64FC1 );
  err = cvCreateMat( 1, count, CV_32FC1 );
  float* err_ptr = err->data.fl;

  if( count > modelPoints )
    {
      ms1 = cvCreateMat( 1, modelPoints, m1->type );
      ms2 = cvCreateMat( 1, modelPoints, m2->type );
    }
  else
    {
      niters = 1;
      ms1 = cvCloneMat(m1);
      ms2 = cvCloneMat(m2);
    }

  //niters = cvRound(log(1-confidence)/log(1-pow(1-outlierRatio,(double)modelPoints)));
  //niters = MIN( MAX(niters, 3), maxIters );

  for( iter = 0; iter < niters; iter++ )
    {
      int i, nmodels;
      if( count > modelPoints )
        {
	  bool found = getSubset( m1, m2, ms1, ms2, 300 );
	  if( !found )
            {
	      if( iter == 0 )
		return false;
	      break;
            }
        }

      nmodels = runKernel( ms1, ms2, models );
      if( nmodels <= 0 )
	continue;
      for( i = 0; i < nmodels; i++ )
        {
	  double SSE = 0;
	  CvMat model_i;
	  cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
	  computeReprojError( m1, m2, &model_i, err );
	  //icvSortDistances( err->data.i, count, 0 );

	  //double median = count % 2 != 0 ?
	  //   err->data.fl[count/2] : (err->data.fl[count/2-1] + err->data.fl[count/2])*0.5; 
	  for(int i2 = 0; i2 < count; i2++)   //Get Sum Squared Error
	    {
	      if(err_ptr[i2] > reprojThreshold)
		SSE += (reprojThreshold*reprojThreshold);
	      else
		SSE += (err_ptr[i2]*err_ptr[i2]);
	    }

	  //cout << "SSE = " << SSE << endl;
	  if( SSE < minSSE )
	    {
	      minSSE = SSE;
	      cvCopy( &model_i, model );
	    }
        }
    }

  if( minSSE < DBL_MAX )
    {
      //sigma = 2.5*1.4826*(1 + 5./(count - modelPoints))*sqrt(minMedian);
      //sigma = MAX( sigma, 0.001 );

      count = findInliers( m1, m2, model, err, mask, reprojThreshold );
      result = count >= modelPoints;
    }

  return result;
}



int estimateTranslation(InputArray _from,
                         OutputArray _out, OutputArray _inliers,
                         double param1, double param2)
{
    Mat from = _from.getMat(), to = _from.getMat();
    int count = from.checkVector(2);

    CV_Assert( count >= 0 );

    _out.create(1, 2, CV_64F);
    Mat out = _out.getMat();

    Mat inliers(1, count, CV_8U);
    inliers = Scalar::all(1);

    Mat dFrom, dTo;
    from.convertTo(dFrom, CV_64F);
    to.convertTo(dTo, CV_64F);
    dFrom = dFrom.reshape(2, 1);
    dTo = dTo.reshape(2, 1);

    CvMat F2x1 = out;
    CvMat mask = inliers;
    CvMat m1 = dFrom;
    CvMat m2 = dTo;

    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < DBL_EPSILON) ? 0.99 : (param2 > 1 - DBL_EPSILON) ? 0.99 : param2;

    //int ok = TranslationEstimator().runRANSAC(&m1, &m2, &F2x1, &mask, param1, param2,2000 );
    //int ok = TranslationEstimator().runLMeDS(&m1, &m2, &F2x1, &mask, param2, 2000 );
    int ok = TranslationEstimator().runMAPSAC(&m1, &m2, &F2x1, &mask, param1, param2,2000 );
    //cout << ok << endl;
    if( _inliers.needed() )
        transpose(inliers, _inliers);

    return ok;
    }

 */ //REMOVED THIS FOR VERSION 3.0.0

/*int main(int argc, char** argv)
{
  std::vector<Point2f> points1(6);
  std::vector<Point2f> points2(6);
  Point2f p1(100,100);
  Point2f p2(5,5);
  Point2f p3(2,2);
  Point2f p4(3,3);
  Point2f p5(4,4);
  Point2f p6(15,15);
  points1[0] = p1;
  points1[1] = p2;
  points1[2] = p3;
  points1[3] = p4;
  points1[4] = p5;
  points1[5] = p6;
  points2[0] = p1;
  points2[1] = p2;
  points2[2] = p3;
  points2[3] = p4;
  points2[4] = p5;
  points2[5] = p6;

  Mat model;
  Mat mask;
  
  int ok = estimateTranslation(points1,points2,model,mask,3,0.9);

  cout << "MODEL: " << model << endl;
  cout << "MASK: " << mask << endl;
  cout << "POINTS1: " << points1 << endl;
  cout << "POINTS2: " << points2 << endl;
  cout << ok << endl;
}
*/


 //----------------------------------------------------------
 // rotation and translation in 2D from point correspondences
 //------------------------------------------------------------
void rigidTransform2D(const int N) {

  // Algorithm: http://igl.ethz.ch/projects/ARAP/svd_rot.pdf

  const bool debug = true;      // print more debug info
  //const bool add_noise = true; // add noise to imput and output
  srand(time(NULL));           // randomize each time

  /*********************************
   * Creat data with some noise
   **********************************/
  float w = 100, h = 100;
  // Simulated transformation
  //Point2f T(1.7f, 17.1f);
  Point2f T(117,17.17);
  float a = 117; // [-180, 180], see atan2(y, x)
  //float noise_level = 0.1f;
  cout<<"True parameters: rot = "<<a<<"deg., T = "<<T << endl;

  // noise
  /*vector<Point2f> noise_src(N), noise_dst(N);
  for (int i=0; i<N; i++) {
    noise_src[i] = Point2f(randf(noise_level), randf(noise_level));
    noise_dst[i] = Point2f(randf(noise_level), randf(noise_level));
    }*/

  // create data with noise
  vector<Point2f> src(N), dst(N);
  float Rdata = 10.0f; // radius of data
  float cosa = cos(a*CV_PI/180);
  float sina = sin(a*CV_PI/180);
  Point2f sonarRotCenter = Point2f(w/2,h);

  //DEBUG: FAKE SRC DATA
  src[0] = Point2f(-10,10);
  src[1] = Point2f(0,10);
  src[2] = Point2f(10,10);
  src[3] = Point2f(-10,-10);
  src[4] = Point2f(-1,-10);
  src[5] = Point2f(-100,-10);
  src[6] = Point2f(100,100);
  src[7] = Point2f(0,100);
  src[8] = Point2f(w/2,h);
  src[9] = Point2f(0,0);

  int N_debug = 10;

  for (int i=0; i<N_debug; i++) {

    // src
    //float x1 = src[i].x;
    //float y1 = src[i].y;
    float x1 = src[i].x - sonarRotCenter.x;
    float y1 = src[i].y - sonarRotCenter.y;
    //src[i] = Point2f(x1,y1);
    //if (add_noise)
    //  src[i] += noise_src[i];

    // dst
    float x2 = x1*cosa - y1*sina;
    float y2 = x1*sina + y1*cosa;
    dst[i] = Point2f(x2,y2) + T;
    dst[i] = dst[i] + sonarRotCenter;//Add back the sonar center
    //if (add_noise)
    //  dst[i] += noise_dst[i];

    if (debug)
      cout<<i<<": "<<src[i]<<"---[" << x1 << "," << y1 << "]---[" << x2 << "," << y2 << "]---" <<dst[i]<<endl;
  }

  // Calculate data centroids
  Scalar centroid_src = mean(src);
  Scalar centroid_dst = mean(dst);
  Point2f center_src(centroid_src[0], centroid_src[1]);
  Point2f center_dst(centroid_dst[0], centroid_dst[1]);
  if (debug)
    cout<<"Centers: "<<center_src<<", "<<center_dst<<endl;

  /*********************************
   * Visualize data
   **********************************/

  // Visualization
  namedWindow("data", 1);
  Mat Mdata(w, h, CV_8UC3); Mdata = Scalar(0);
  Point2f center_img(w/2, h/2);

  float scl = 0.4*min(w/Rdata, h/Rdata); // compensate for noise
  scl/=sqrt(2); // compensate for rotation effect
  Point2f dT = (center_src+center_dst)*0.5; // compensate for translation

  for (int i=0; i<N; i++) {
    Point2f p1(scl*(src[i] - dT));
    Point2f p2(scl*(dst[i] - dT));
    // invert Y axis
    p1.y = -p1.y; p2.y = -p2.y;
    // add image center
    p1+=center_img; p2+=center_img;
    circle(Mdata, p1, 1, Scalar(0, 255, 0));
    circle(Mdata, p2, 1, Scalar(0, 0, 255));
    line(Mdata, p1, p2, Scalar(100, 100, 100));

  }

  /*********************************
   * Get 2D rotation and translation
   **********************************/

  //markTime();

  // subtract centroids from data
  for (int i=0; i<N; i++) {
    src[i] -= center_src;
    dst[i] -= center_dst;
  }

  // compute a covariance matrix
  float Cxx = 0.0, Cxy = 0.0, Cyx = 0.0, Cyy = 0.0;
  for (int i=0; i<N; i++) {
    Cxx += src[i].x*dst[i].x;
    Cxy += src[i].x*dst[i].y;
    Cyx += src[i].y*dst[i].x;
    Cyy += src[i].y*dst[i].y;
  }
  Mat Mcov = (Mat_<float>(2, 2)<<Cxx, Cxy, Cyx, Cyy);
  if (debug)
    cout<<"Covariance Matrix "<<Mcov<<endl;

  // SVD
  cv::SVD svd;
  svd = SVD(Mcov, SVD::FULL_UV);
  if (debug) {
    cout<<"U = "<<svd.u<<endl;
    cout<<"W = "<<svd.w<<endl;
    cout<<"V transposed = "<<svd.vt<<endl;
  }

  // rotation = V*Ut
  Mat V = svd.vt.t();
  Mat Ut = svd.u.t();
  float det_VUt = determinant(V*Ut);
  Mat W = (Mat_<float>(2, 2)<<1.0, 0.0, 0.0, det_VUt);
  float rot[4];
  Mat R_est(2, 2, CV_32F, rot);
  R_est = V*W*Ut;
  if (debug)
    cout<<"Rotation matrix: "<<R_est<<endl;

  float cos_est = rot[0];
  float sin_est = rot[2];
  float ang = atan2(sin_est, cos_est);

  // translation = mean_dst - R*mean_src
  Point2f center_srcSonar = center_src - sonarRotCenter; //DEBUG AMS added this
  //  Point2f center_srcRot = Point2f(
  //				  cos_est*center_src.x - sin_est*center_src.y,
  //				  sin_est*center_src.x + cos_est*center_src.y);
  Point2f center_srcRot = Point2f(
				  cos_est*center_srcSonar.x - sin_est*center_srcSonar.y,
				  sin_est*center_srcSonar.x + cos_est*center_srcSonar.y);
  center_srcRot += sonarRotCenter; //DEBUG AMS ADDED
  Point2f T_est = center_dst - center_srcRot;

  cout << "Center src: " << endl << center_src << endl;
  cout << "sonar Rot Center: " << sonarRotCenter << endl;
  cout << "Center src Sonar: " << endl << center_srcSonar << endl;
  cout << "Center src Rot: " << endl << center_srcRot << endl;
  cout << "Center dst: " << endl << center_dst << endl;

  // RMSE
  double RMSE = 0.0;
  for (int i=0; i<N; i++) {
    Point2f dst_est(
		    cos_est*src[i].x - sin_est*src[i].y,
		    sin_est*src[i].x + cos_est*src[i].y);
    RMSE += (dst[i].x - dst_est.x)*(dst[i].x - dst_est.x) + (dst[i].y - dst_est.y)*(dst[i].y - dst_est.y);
  }
  if (N>0)
    RMSE = sqrt(RMSE/N);

  // Final estimate msg
  cout<<"Estimate = "<<ang*180/CV_PI<<"deg., T = "<<T_est<<"; RMSE = "<<RMSE<<endl;

  // show image
  //printTime(1);
  imshow("data", Mdata);
  waitKey(-1);

  return;
} // rigidTransform2D()
