#import <opencv2/opencv.hpp>
#import "OpenCVWrapper.h"

using namespace std;

@implementation OpenCVWrapper 
static CGRect mrzRect;
static NSString* mrzText;
+(cv::Mat)cvMatFromUIImage:(UIImage *)image
{
  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
  CGFloat cols = image.size.width;
  CGFloat rows = image.size.height;
  cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                 cols,                       // Width of bitmap
                                                 rows,                       // Height of bitmap
                                                 8,                          // Bits per component
                                                 cvMat.step[0],              // Bytes per row
                                                 colorSpace,                 // Colorspace
                                                 kCGImageAlphaNoneSkipLast |
                                                 kCGBitmapByteOrderDefault); // Bitmap info flags
  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
  CGContextRelease(contextRef);
  return cvMat;
}
+(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
  NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
  CGColorSpaceRef colorSpace;
  if (cvMat.elemSize() == 1) {
      colorSpace = CGColorSpaceCreateDeviceGray();
  } else {
      colorSpace = CGColorSpaceCreateDeviceRGB();
  }
  CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
  // Creating CGImage from cv::Mat
  CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                     cvMat.rows,                                 //height
                                     8,                                          //bits per component
                                     8 * cvMat.elemSize(),                       //bits per pixel
                                     cvMat.step[0],                            //bytesPerRow
                                     colorSpace,                                 //colorspace
                                     kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                     provider,                                   //CGDataProviderRef
                                     NULL,                                       //decode
                                     false,                                      //should interpolate
                                     kCGRenderingIntentDefault                   //intent
                                     );
  // Getting UIImage from CGImage
  UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
  CGImageRelease(imageRef);
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(colorSpace);
  return finalImage;
 }

+(UIImage*)readMRZ:(UIImage*)image{
    cv::Mat rgbMat = [OpenCVWrapper cvMatFromUIImage:image];
    int rows = rgbMat.rows;
    int cols = rgbMat.cols;
    
    cv::Mat greyMat;
    cv::cvtColor(rgbMat, greyMat, cv::COLOR_BGR2GRAY);
    
    //Binary image
    cv::Mat binaryMat(greyMat.size(), greyMat.type());

    //Apply thresholding
    cv::threshold(greyMat, binaryMat, 100, 255, cv::THRESH_BINARY);
    
    cv::Mat invertMat(binaryMat.size(), binaryMat.type());
    bitwise_not ( binaryMat, invertMat );
    
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement( cv::MORPH_ELLIPSE,
    cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
    cv::Point( dilation_size, dilation_size ) );
    
    cv::Mat closed;
    morphologyEx(invertMat,closed, cv::MORPH_CLOSE, kernel);
    
    
    // find contours and store them all as a list
    vector<vector<cv::Point> > contours;
    findContours(closed, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    
    vector<cv::Point> approx;
    
    // test each contour
    for( size_t i = 0; i < contours.size(); i++ )
    {
        // approximate contour with accuracy proportional
        // to the contour perimeter
        approxPolyDP(cv::Mat(contours[i]), approx, arcLength(cv::Mat(contours[i]), true)*0.02, true);

        // square contours should have 4 vertices after approximation
        if( approx.size() == 4){
            cv::Rect rect = boundingRect(contours[i]);
            if(rect.width>0.5*cols && rect.width<0.9*cols && rect.height>0.03*rows && rect.height<0.4*rows){
                //printf("[%d,%d,%d,%d]\n",rect.x,rect.y,rect.width,rect.height);
                int padding=15;
                rect.x = rect.x - padding;
                rect.y = rect.y - padding;
                rect.width = rect.width + 2*padding;
                rect.height = rect.height + 2*padding;
                if(rect.x>=0 && rect.y>=0 && rect.x+rect.width<=cols && rect.y+rect.height<=rows){
                    cv :: Mat image_roi;
                    cv::Rect roi(rect.x, rect.y, rect.width, rect.height);
                    cv::Mat(rgbMat, roi).copyTo(image_roi);
                    cv::resize(image_roi, image_roi, cv::Size(image_roi.cols * 5.0,image_roi.rows * 5.0), 0, 0, CV_INTER_LINEAR);
                    mrzRect = CGRectMake(rect.y,rect.x, 1.5*rect.height,1.5*rect.width);
                    cv::Mat preprocessed = [OpenCVWrapper preprocessMRZForOCR:image_roi];
                    mrzText = [OpenCVWrapper getText:preprocessed];
                    cv::rectangle(rgbMat, rect, cv::Scalar(0, 255, 0));
                    return [OpenCVWrapper UIImageFromCVMat:image_roi];
                    
                }
                
            }
            
        }
    }
    
    return nil;
}

+(CGRect)getMRZRect{
    return mrzRect;
}

+(NSString*)getText:(cv::Mat)mat{
    UIImage* image = [OpenCVWrapper UIImageFromCVMat:mat];
    G8Tesseract *tesseract = [[G8Tesseract alloc] initWithLanguage:@"eng+ita"];
    tesseract.delegate = self;
    tesseract.charWhitelist = @"0123456789<ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    
    tesseract.image = image;
    BOOL flag = [tesseract recognize];
    NSString* text=@"";
    if(flag){
        text = [tesseract recognizedText];
    }
    return text;
}

+(cv::Mat)preprocessMRZForOCR:(cv::Mat)mat{
    cv::fastNlMeansDenoisingColored( mat, mat, 10, 10, 7, 15 );
    UIImage* image = [OpenCVWrapper UIImageFromCVMat:mat];
    
    cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
    image = [OpenCVWrapper UIImageFromCVMat:mat];
    
    cv::fastNlMeansDenoising(mat, mat, 3.0, 7, 21);
    image = [OpenCVWrapper UIImageFromCVMat:mat];
    
    cv::threshold(mat, mat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    image = [OpenCVWrapper UIImageFromCVMat:mat];
    
    return mat;
}

+(NSString*)getMRZText{
    return mrzText;
}

- (void)progressImageRecognitionForTesseract:(G8Tesseract *)tesseract {
    NSLog(@"progress: %lu", (unsigned long)tesseract.progress);
}

- (BOOL)shouldCancelImageRecognitionForTesseract:(G8Tesseract *)tesseract {
    return NO;  // return YES, if you need to interrupt tesseract before it finishes
}

@end
