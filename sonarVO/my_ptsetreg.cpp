//#include "/home/anthony/opencv_2.3.0/opencv/modules/calib3d/src/precomp.hpp"
#include "precomp.hpp"
#include "my_ptsetreg.hpp"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <limits>
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

//--------------------------------------------------------------------
//AMS ADDED NEW MAPSAC CODE 7/10/2014

//namespace cv
//{
  int MAPSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters )
  { //SHOULDN"T BE USED WITH MAPSAC - ONLY RANSAC??
    if( modelPoints <= 0 )
      CV_Error( Error::StsOutOfRange, "the number of model points should be positive" );
    
    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);
    
    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - std::pow(1. - ep, modelPoints);
    if( denom < DBL_MIN )
      return 0;
    
    num = std::log(num);
    denom = std::log(denom);
    
    return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num/denom);
  }
  
/*  
class MAPSACPointSetRegistrator //: public PointSetRegistrator //DEBUG AMS
{
  public:
    MAPSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb=Ptr<PointSetRegistrator::Callback>(),
			      int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000)
      : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters)
    {
      checkPartialSubsets = true;
    }
*/ 
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
    /*
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
		  cout << i2 << " --- " << errptr[i2] << " += " << SSE << endl;
		}

	      cout << "SSE = " << SSE << endl;

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
	  cout << "MIN_SSE = " << minSSE << endl;
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
  
  
  //CV_INIT_ALGORITHM(MAPSACPointSetRegistrator, "PointSetRegistrator.MAPSAC", 
  //		    obj.info()->addParam(obj, "threshold", obj.threshold); 
  //		    obj.info()->addParam(obj, "confidence", obj.confidence); 
  //		    obj.info()->addParam(obj, "maxIters", obj.maxIters))
  */
  Ptr<MAPSACPointSetRegistrator> createMAPSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb, int _modelPoints, double _threshold,  double _confidence, int _maxIters)
  {
    //CV_Assert( !MAPSACPointSetRegistrator_info_auto.name().empty() );
    return Ptr<MAPSACPointSetRegistrator>(
				    new MAPSACPointSetRegistrator(_cb, _modelPoints, _threshold, _confidence, _maxIters));
				    }
//}
  
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

    /*

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

*/
int estimateTranslationNew(InputArray _from, InputArray _to, int _method,
                         OutputArray _out, OutputArray _inliers,
                         double param1, double param2)
{
    Mat from = _from.getMat(), to = _to.getMat();
    int count = from.checkVector(2);
    
    CV_Assert( count >= 0 && to.checkVector(2) == count );

    Mat dFrom, dTo;
    from.convertTo(dFrom, CV_64F);
    to.convertTo(dTo, CV_64F);
    dFrom = dFrom.reshape(2, count);
    dTo = dTo.reshape(2, count);

    const double epsilon = DBL_EPSILON;
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;
    int _maxIters = 1000;
    //cout << "Method = " << _method << ", RANSAC=" << RANSAC << ", MAPSAC=" << MAPSAC << ", LMEDS=" << LMEDS << endl;
    //DEBUG
    if(_method == RANSAC)
      return createRANSACPointSetRegistrator(makePtr<TranslationEstimatorCallback>(), 1, param1, param2)->run(dFrom, dTo, _out, _inliers);
    else if(_method == MAPSAC)
      return createMAPSACPointSetRegistrator(makePtr<TranslationEstimatorCallback>(), 1, param1, param2,_maxIters)->run(dFrom, dTo, _out, _inliers);
    else //LMEDS
      return createLMeDSPointSetRegistrator(makePtr<TranslationEstimatorCallback>(), 1, param2)->run(dFrom, dTo, _out, _inliers);
}

void testEstimateTranslationNew()
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
  /*
  cv::Ptr<CvMat> points1_mat = cvCreateMat(points1.size(),2, CV_32F);
  cv::Ptr<CvMat> points2_mat = cvCreateMat(points2.size(),2, CV_32F);
 
  float* ptr1 = reinterpret_cast<float*>(points1_mat->data.ptr); 
  float* ptr2 = reinterpret_cast<float*>(points2_mat->data.ptr); 
  float* ptrVec = reinterpret_cast<float*>(points1.data());

  //ptr1[0] = p1.x;
  //cout << points1 << " " << ptr1[0] << " "  << endl;

    for(int i = 0; i<6; i += 2)
    {
      ptr1[i] = points1[i].x;
      ptr1[i+1] = points1[i].y;
      cout << ptr1[i] << "," << ptr1[i+1] << endl;
    }

  cout << "VEC = " << points1 << endl;
  //cout << "MAT = " << points1_mat << endl;

  cv::Ptr<CvMat> model = cvCreateMat(2,1,CV_32F);
  */
  //cv::TranslationEstimator().runKernel(points1_mat,points2_mat,model);
  

  Mat model;
  Mat mask;
  int robustEstMeth = RANSAC; //RANSAC or MAPSAC or LMEDS

  //cout << "POINTS 1: " << points1 << endl;

  int ok = estimateTranslationNew(points1,points1,robustEstMeth,model,mask,3,0.9);

  std::cout << "MODEL: " << model << endl;
  std::cout << "MASK: " << mask << endl;
  std::cout << "POINTS1: " << points1 << endl;
  std::cout << "POINTS2: " << points2 << endl;
  std::cout << ok << endl;
}

//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------

int estimateRigidTransform2DNew(InputArray _from, InputArray _to, int _method,
                         OutputArray _out, OutputArray _inliers,
				double param1, double param2, Size _imgSize)
{
    Mat from = _from.getMat(), to = _to.getMat();
    int count = from.checkVector(2);
   
    CV_Assert( count >= 0 && to.checkVector(2) == count );

    Mat dFrom, dTo;
    from.convertTo(dFrom, CV_64F);
    to.convertTo(dTo, CV_64F);
    dFrom = dFrom.reshape(2, count);
    dTo = dTo.reshape(2, count);

    const double epsilon = DBL_EPSILON;
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;
    int _maxIters = 1000;
    //DEBUG
    //return createRANSACPointSetRegistrator(makePtr<TranslationEstimatorCallback>(), 1, param1, param2)->run(dFrom, dTo, _out, _inliers);
    Ptr<RigidTransform2DEstimatorCallback> rtec = makePtr<RigidTransform2DEstimatorCallback>();
    rtec->imgWidth = _imgSize.width;
    rtec->imgHeight = _imgSize.height;
    int modelPoints = 3;
    //cout << "Robust Est Method = " << _method << ", RANSAC=" << RANSAC << ", MAPSAC=" << MAPSAC << ", LMEDS=" << LMEDS << endl;
    //cout << param1 << ", Param2 = " << param2 << endl;
    if(_method == RANSAC)
      return createRANSACPointSetRegistrator(rtec, modelPoints, param1, param2,_maxIters)->run(dFrom, dTo, _out, _inliers);
    else if(_method == MAPSAC) 
      return createMAPSACPointSetRegistrator(rtec, modelPoints, param1, param2)->run(dFrom, dTo, _out, _inliers);
    else //LMEDS
      return createLMeDSPointSetRegistrator(rtec, modelPoints, param2)->run(dFrom, dTo, _out, _inliers);
    //return createMAPSACPointSetRegistrator(makePtr<RigidTransform2DEstimatorCallback>(), 1, param1, param2,_maxIters)->run(dFrom, dTo, _out, _inliers);
}



void testEstimateRigidTransform2DNew()
{
  std::vector<Point2f> points1(6);
  std::vector<Point2f> points2(6);
  Point2f p1(100,100);
  Point2f p2(0,100);
  Point2f p3(0,0);
  Point2f p4(3,300);
  Point2f p5(4,10);
  Point2f p6(15,-100);
  points1[0] = p1;
  points1[1] = p2;
  points1[2] = p3;
  points1[3] = p4;
  points1[4] = p5;
  points1[5] = p6;

  float w = 400, h = 200;
  Point2f T(17.7, 111.7);
  float angleDeg = -117;
  float cosa = cos(angleDeg*CV_PI/180);
  float sina = sin(angleDeg*CV_PI/180);
  Point2f sonarRotCenter(w/2, h);

  int N_debug = 6;
  for(int i=0; i<N_debug; i++)
    {
      float x1 = points1[i].x - sonarRotCenter.x;
      float y1 = points1[i].y - sonarRotCenter.y;

      float x2 = x1*cosa - y1*sina;
      float y2 = x1*sina + y1*cosa;
      points2[i] = Point2f(x2,y2) + T + sonarRotCenter;
    }
  
  Mat model;
  Mat mask;
  Size imgSize(w,h);
  int robustEstMeth = RANSAC; // RANSAC or MAPSAC or LMEDS

  //cout << "POINTS 1: " << points1 << endl;

  int ok = estimateRigidTransform2DNew(points1,points2,robustEstMeth,model,mask,3,0.9,imgSize);

  std::cout << "MODEL: " << model << endl;
  std::cout << "MASK: " << mask << endl;
  std::cout << "POINTS1: " << points1 << endl;
  std::cout << "POINTS2: " << points2 << endl;
  std::cout << ok << endl;
}
