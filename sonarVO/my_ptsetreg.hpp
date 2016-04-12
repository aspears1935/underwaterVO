#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "precomp.hpp"

#define MAPSAC 7
#define DEBUG false

int MAPSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters );
/*
class MAPSACPointSetRegistrator : public Algorithm
{
public:
  MAPSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb=Ptr<PointSetRegistrator::Callback>(), int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000)  : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters)
{
  checkPartialSubsets = true;
};*/
  /*
    class Callback
  {
  public:
    virtual ~Callback() {}
    virtual int runKernel(InputArray m1, InputArray m2, OutputArray model) const = 0;
    virtual void computeError(InputArray m1, InputArray m2, InputArray model, OutputArray err) const = 0;
    virtual bool checkSubset(InputArray, InputArray, int) const { return true; }
  };
  */
    /*
  void setCallback(const Ptr<PointSetRegistrator::Callback>& cb);
  bool run(InputArray m1, InputArray m2, OutputArray model, OutputArray mask);

  AlgorithmInfo* info() const;
  Ptr<PointSetRegistrator::Callback> cb;
  int modelPoints;
  bool checkPartialSubsets;
  double threshold;
  double confidence;
  int maxIters;

};
    */

class MAPSACPointSetRegistrator //: public PointSetRegistrator //DEBUG AMS
{
  public:
    MAPSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb=Ptr<PointSetRegistrator::Callback>(),
			      int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000)
      : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters)
    {
      checkPartialSubsets = true;
    }
    
  //AMS ADDED:
  /*   class Callback
   {
   public:
     virtual ~Callback() {}
     virtual int runKernel(InputArray m1, InputArray m2, OutputArray model) const = 0;
     virtual void computeError(InputArray m1, InputArray m2, InputArray model, OutputArray err) const = 0;
     virtual bool checkSubset(InputArray, InputArray, int) const { return true; }
   };
  */

    int findInliers( const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh ) const
    {
      cb->computeError( m1, m2, model, err );
      mask.create(err.size(), CV_8U);
      
      CV_Assert( err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
      const float* errptr = err.ptr<float>();
      uchar* maskptr = mask.ptr<uchar>();
      float t = (float)(thresh*thresh);
      int i, n = (int)err.total(), nz = 0;
      for( i = 0; i < n; i++ )
	{
	  int f = errptr[i] <= t;
	  maskptr[i] = (uchar)f;
	  nz += f;
	}
      return nz;
    }
    
    bool getSubset( const Mat& m1, const Mat& m2,
		    Mat& ms1, Mat& ms2, RNG& rng,
		    int maxAttempts=1000 ) const
    {
      cv::AutoBuffer<int> _idx(modelPoints);
      int* idx = _idx;
      int i = 0, j, k, iters = 0;
      int esz1 = (int)m1.elemSize(), esz2 = (int)m2.elemSize();
      int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
      int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
      int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
      const int *m1ptr = (const int*)m1.data, *m2ptr = (const int*)m2.data;
      
      ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
      ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));
      
      int *ms1ptr = (int*)ms1.data, *ms2ptr = (int*)ms2.data;
      
      CV_Assert( count >= modelPoints && count == count2 );
      CV_Assert( (esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0 );
      esz1 /= sizeof(int);
      esz2 /= sizeof(int);
      
      for(; iters < maxAttempts; iters++)
	{
	  for( i = 0; i < modelPoints && iters < maxAttempts; )
	    {
	      int idx_i = 0;
	      for(;;)
		{
		  idx_i = idx[i] = rng.uniform(0, count);
		  for( j = 0; j < i; j++ )
		    if( idx_i == idx[j] )
		      break;
		  if( j == i )
		    break;
		}
	      for( k = 0; k < esz1; k++ )
		ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
	      for( k = 0; k < esz2; k++ )
		ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
	      if( checkPartialSubsets && !cb->checkSubset( ms1, ms2, i+1 ))
		{
		  iters++;
		  continue;
		}
	      i++;
	    }
	  if( !checkPartialSubsets && i == modelPoints && !cb->checkSubset(ms1, ms2, i))
	    continue;
	  break;
	}
      
      return i == modelPoints && iters < maxAttempts;
    }
    
  bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const
    {
      bool result = false;
      Mat m1 = _m1.getMat(), m2 = _m2.getMat();
      Mat err, mask, model, bestModel, ms1, ms2;
      
      int iter, niters = MAX(maxIters, 1);
      int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
      int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
      int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;
      double minSSE = DBL_MAX; //AMS

      RNG rng((uint64)-1);
      
      CV_Assert( cb );
      CV_Assert( confidence > 0 && confidence < 1 );
      
      CV_Assert( count >= 0 && count2 == count );
      if( count < modelPoints )
	return false;
      
      Mat bestMask0, bestMask;
      
      if( _mask.needed() )
	{
	  _mask.create(count, 1, CV_8U, -1, true);
	  bestMask0 = bestMask = _mask.getMat();
	  CV_Assert( (bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count );
	}
      else
	{
	  bestMask.create(count, 1, CV_8U);
	  bestMask0 = bestMask;
	}
      
      if( count == modelPoints )
	{
	  if( cb->runKernel(m1, m2, bestModel) <= 0 )
	    return false;
	  bestModel.copyTo(_model);
	  bestMask.setTo(Scalar::all(1));
	  return true;
	}
      
      for( iter = 0; iter < niters; iter++ )
	{
	  int i, goodCount, nmodels;
	  if( count > modelPoints )
	    {
	      bool found = getSubset( m1, m2, ms1, ms2, rng );
	      if( !found )
		{
		  if( iter == 0 )
		    return false;
		  break;
		}
	    }
	  
	  nmodels = cb->runKernel( ms1, ms2, model );
	  if( nmodels <= 0 )
	    continue;
	  CV_Assert( model.rows % nmodels == 0 );
	  Size modelSize(model.cols, model.rows/nmodels);
	  
	  for( i = 0; i < nmodels; i++ )
	    {
	      double SSE = 0; //AMS
	      Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
	      
	      //Compute Reproj Error
	      goodCount = findInliers( m1, m2, model_i, err, mask, threshold );
	      const float* errptr = err.ptr<float>();

	      //Compute Sum Squared Err for MAPSAC
	      for(int i2=0; i2<count; i2++)
		{
		  if(errptr[i2] > threshold)
		    SSE += (threshold*threshold); //MAPSAC
		  else
		    SSE += (errptr[i2]*errptr[i2]);
		  //cout << i2 << " --- " << errptr[i2] << " += " << SSE << endl;
		}

	      //cout << "SSE = " << SSE << endl;

	      if( SSE < minSSE )                   
                {                                       
                  std::swap(mask, bestMask);                            
                  model_i.copyTo(bestModel);                                  
                  maxGoodCount = goodCount;
		  minSSE = SSE; //AMS
		    //niters = MAPSACUpdateNumIters( confidence, (double)(count - goodCount)/count, modelPoints, niters ); //MAPSAC isn't based on this idea??    
                }                                                   
            }                                       
        }                          
      
      if( minSSE < DBL_MAX )                               
        {                                                           
          if( bestMask.data != bestMask0.data ) 
            {                
              if( bestMask.size() == bestMask0.size() ) 
                bestMask.copyTo(bestMask0);      
              else              
                transpose(bestMask, bestMask0);     
            }                                       
          bestModel.copyTo(_model); 
	  //maxgoodCount = findInliers( m1, m2, model_i, err, mask, threshold );
          result = maxGoodCount >= modelPoints;
	  //cout << "MIN_SSE = " << minSSE << endl;
        }                                      
      else                                           
      _model.release();
      
      return result;
    }
    
    void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) { cb = _cb; }
    
    AlgorithmInfo* info() const;
    
    Ptr<PointSetRegistrator::Callback> cb;
    int modelPoints;
    bool checkPartialSubsets;
    double threshold;
    double confidence;
    int maxIters;
  };
  
Ptr<MAPSACPointSetRegistrator> createMAPSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& cb, int modelPoints, double threshold, double confidence=0.99, int maxIters=1000 );


//-------------------------------------------------------------------

/*
class TranslationEstimatorCallback : public PointSetRegistrator::Callback
{
public:
  int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const;
  void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const;
  bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const;
};
*/

class TranslationEstimatorCallback : public PointSetRegistrator::Callback
{
public:
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();

        const Point2d* from = m1.ptr<Point2d>();
        const Point2d* to   = m2.ptr<Point2d> (); //WE DONT USE THIS.

	Mat modelMat = Mat::zeros(1,2,CV_64F);

	modelMat.at<double>(0,0) = from[0].x;
	modelMat.at<double>(0,1) = from[0].y;
	
	//cout << "modelMat=" << modelMat << endl;

	modelMat.copyTo(_model);
	Mat model = _model.getMat();

	//cout << "m1 in runKernel = " << m1 << endl;
	//cout << "MODEL from runKernel = " << model << endl;

        return 1;
    }

    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        const Point2d* from = m1.ptr<Point2d>();
        const Point2d* to   = m2.ptr<Point2d>();
        const double* modelPtr = model.ptr<double>();

        int count = m1.checkVector(2);
        CV_Assert( count > 0 );

        _err.create(count, 1, CV_32F);
        Mat err = _err.getMat();
        float* errptr = err.ptr<float>();

        for(int i = 0; i < count; i++ )
        {
            const Point2d& f = from[i];
            //const Point2d& t = to[i];

	    double xshift = f.x - modelPtr[0];
	    double yshift = f.y - modelPtr[1];

	    //cout << "ERROR:" << xshift << "," << yshift << endl;

            errptr[i] = (float)std::sqrt(xshift*xshift + yshift*yshift);
        }
    }

    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
    {
        return true;
    }
};

  int estimateTranslationNew(InputArray _from, InputArray _to, int _method,
			     OutputArray _out, OutputArray _inliers,
			     double param1, double param2);

void testEstimateTranslationNew();

  //-----------------------------------------------------------------
  //-----------------------------------------------------------------
  //-----------------------------------------------------------------

class RigidTransform2DEstimatorCallback : public PointSetRegistrator::Callback
{
public:
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
      
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
	Point2d sonarRotCenter(imgWidth/2,imgHeight);

	//cout << "SONAR ROT CENTER = " << sonarRotCenter << endl;

        const Point2d* from = m1.ptr<Point2d>();
        const Point2d* to   = m2.ptr<Point2d>();

	//cout << "m1 = " << m1 << endl;
	//cout << "m2 = " << m2 << endl;

	Mat modelMat = Mat::zeros(1,3,CV_64F);

	Scalar centroid_src = mean(m1);
	Scalar centroid_dst = mean(m2);
	Point2d center_src(centroid_src[0], centroid_src[1]);
	Point2d center_dst(centroid_dst[0], centroid_dst[1]);

	int count = m1.checkVector(2);
	//cout << "count (should be 3) = " << count << endl;

	vector<Point2d> fromSubMean(count);
	vector<Point2d> toSubMean(count);	

	// subtract centroids from data
	for (int i=0; i<count; i++) {
	  fromSubMean[i] = from[i] - center_src;
	  toSubMean[i] = to[i] - center_dst;
	}

	// compute a covariance matrix         
	double Cxx = 0.0, Cxy = 0.0, Cyx = 0.0, Cyy = 0.0;
	for (int i=0; i<count; i++) {
	  Cxx += fromSubMean[i].x*toSubMean[i].x;
	  Cxy += fromSubMean[i].x*toSubMean[i].y;
	  Cyx += fromSubMean[i].y*toSubMean[i].x;
	  Cyy += fromSubMean[i].y*toSubMean[i].y;
	}

	Mat Mcov = (Mat_<double>(2, 2)<<Cxx, Cxy, Cyx, Cyy);
	if (DEBUG)
	  cout<<"Covariance Matrix "<<Mcov<<endl;

	// SVD
	cv::SVD svd;
	svd = SVD(Mcov, SVD::FULL_UV);
	if (DEBUG) {
	  cout<<"U = "<<svd.u<<endl;
	  cout<<"W = "<<svd.w<<endl;
	  cout<<"V transposed = "<<svd.vt<<endl;
	}

	// rotation = V*Ut
	Mat V = svd.vt.t();
	Mat Ut = svd.u.t();
	double det_VUt = determinant(V*Ut);
	Mat W = (Mat_<double>(2, 2)<<1.0, 0.0, 0.0, det_VUt);
	double rot[4];
	Mat R_est(2, 2, CV_64F, rot);
	R_est = V*W*Ut;
	//cout<<"Rotation matrix: "<<R_est<<endl;

	double cos_est = rot[0];
	double sin_est = rot[2];
	double ang = atan2(sin_est, cos_est);

	//cout << "Est Angle = " << ang*180/CV_PI << endl;

	// translation = mean_dst - R*mean_src
	Point2d center_srcSonar = center_src - sonarRotCenter; //DEBUG AMS added this
	//  Point2d center_srcRot = Point2d(
	//  cos_est*center_src.x - sin_est*center_src.y,
	//  sin_est*center_src.x + cos_est*center_src.y);
	Point2d center_srcRot = Point2d(
			cos_est*center_srcSonar.x - sin_est*center_srcSonar.y,
			sin_est*center_srcSonar.x + cos_est*center_srcSonar.y);
	center_srcRot += sonarRotCenter; //DEBUG AMS ADDED
	Point2d T_est = center_dst - center_srcRot;
       
	if(DEBUG)
	  {
	    cout << "Center src: " << endl << center_src << endl;
	    cout << "sonar Rot Center: " << sonarRotCenter << endl;
	    cout << "Center src Sonar: " << endl << center_srcSonar << endl;
	    cout << "Center src Rot: " << endl << center_srcRot << endl;
	    cout << "Center dst: " << endl << center_dst << endl;
	  }

	modelMat.at<double>(0,0) = T_est.x;
	modelMat.at<double>(0,1) = T_est.y;
	modelMat.at<double>(0,2) = ang;

	if(DEBUG)
	  cout << "modelMat=" << modelMat << endl;

	modelMat.copyTo(_model);
	Mat model = _model.getMat();

	//cout << "m1 in runKernel = " << m1 << endl;
	//cout << "MODEL from runKernel = " << model << endl;
	
        return 1;
    }

    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        const Point2d* from = m1.ptr<Point2d>();
        const Point2d* to   = m2.ptr<Point2d>();
        const double* modelPtr = model.ptr<double>();

        int count = m1.checkVector(2);
        CV_Assert( count > 0 );

        _err.create(count, 1, CV_32F);
        Mat err = _err.getMat();
        float* errptr = err.ptr<float>();

	Point2d sonarRotCenter(imgWidth/2,imgHeight);
	double xtrans = modelPtr[0];
	double ytrans = modelPtr[1];
	double angle = modelPtr[2]; //RADIANS??
	//cout << "ANGLE(RADS) = " << angle << endl;
	double cos_est = cos(angle);
	double sin_est = sin(angle);
	if(DEBUG)
	  cout << "cos_est, sin_est = " << cos_est << ", " << sin_est << endl;

        for(int i = 0; i < count; i++ )
        {
	  // const Point2d& f = from[i];
            //const Point2d& t = to[i];
	 
	  //NEED TO CHECK THIS. ADD xtrans at end:?????
	  Point2d fSubSon = from[i] - sonarRotCenter;
	  Point2d dst_est(cos_est*fSubSon.x - sin_est*fSubSon.y, 
			  sin_est*fSubSon.x + cos_est*fSubSon.y);
	  dst_est += sonarRotCenter;
	  dst_est += Point2d(xtrans,ytrans);

	  double xdiff = to[i].x - dst_est.x;
	  double ydiff = to[i].y - dst_est.y;

	  //cout << "ERROR:" << xdiff << "," << ydiff << endl;

	  errptr[i] = (float)std::sqrt(xdiff*xdiff + ydiff*ydiff);
        }
    }

    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
    {
        return true;
    }

  double imgWidth;
  double imgHeight;
};

  int estimateRigidTransform2DNew(InputArray _from, InputArray _to, int _model,
			     OutputArray _out, OutputArray _inliers,
				double param1, double param2, Size _imgSize);

void testEstimateRigidTransform2DNew();
