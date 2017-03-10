package it.polito.teaching.cv;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import it.polito.elite.teaching.cv.utils.Utils;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

/**
 * The controller associated to the only view of our application. The
 * application logic is implemented here. It handles the button for opening an
 * image and perform all the operation related to the Fourier transformation and
 * antitransformation.
 * 
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 2.0 (2017-03-10)
 * @since 1.0 (2013-12-11)
 * 
 */
public class FourierController
{
	// images to show in the view
	@FXML
	private ImageView originalImage;
	@FXML
	private ImageView transformedImage;
	@FXML
	private ImageView antitransformedImage;
	// a FXML button for performing the transformation
	@FXML
	private Button transformButton;
	// a FXML button for performing the antitransformation
	@FXML
	private Button antitransformButton;
	
	// the main stage
	private Stage stage;
	// the JavaFX file chooser
	private FileChooser fileChooser;
	// support variables
	private Mat image;
	private List<Mat> planes;
	// the final complex image
	private Mat complexImage;
	
	/**
	 * Init the needed variables
	 */
	protected void init()
	{
		this.fileChooser = new FileChooser();
		this.image = new Mat();
		this.planes = new ArrayList<>();
		this.complexImage = new Mat();
	}
	
	/**
	 * Load an image from disk
	 */
	@FXML
	protected void loadImage()
	{
		// show the open dialog window
		File file = this.fileChooser.showOpenDialog(this.stage);
		if (file != null)
		{
			// read the image in gray scale
			this.image = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
			// show the image
			this.updateImageView(originalImage, Utils.mat2Image(this.image));
			// set a fixed width
			this.originalImage.setFitWidth(250);
			// preserve image ratio
			this.originalImage.setPreserveRatio(true);
			// update the UI
			this.transformButton.setDisable(false);
			
			// empty the image planes and the image views if it is not the first
			// loaded image
			if (!this.planes.isEmpty())
			{
				this.planes.clear();
				this.transformedImage.setImage(null);
				this.antitransformedImage.setImage(null);
			}
			
		}
	}
	
	/**
	 * The action triggered by pushing the button for apply the dft to the
	 * loaded image
	 */
	@FXML
	protected void transformImage()
	{
		// optimize the dimension of the loaded image
		Mat padded = this.optimizeImageDim(this.image);
		padded.convertTo(padded, CvType.CV_32F);
		// prepare the image planes to obtain the complex image
		this.planes.add(padded);
		this.planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
		// prepare a complex image for performing the dft
		Core.merge(this.planes, this.complexImage);
		
		// dft
		Core.dft(this.complexImage, this.complexImage);
		
		// optimize the image resulting from the dft operation
		Mat magnitude = this.createOptimizedMagnitude(this.complexImage);
		
		// show the result of the transformation as an image
		this.updateImageView(transformedImage, Utils.mat2Image(magnitude));
		// set a fixed width
		this.transformedImage.setFitWidth(250);
		// preserve image ratio
		this.transformedImage.setPreserveRatio(true);
		
		// enable the button for performing the antitransformation
		this.antitransformButton.setDisable(false);
		// disable the button for applying the dft
		this.transformButton.setDisable(true);
	}
	
	/**
	 * The action triggered by pushing the button for apply the inverse dft to
	 * the loaded image
	 */
	@FXML
	protected void antitransformImage()
	{
		Core.idft(this.complexImage, this.complexImage);
		
		Mat restoredImage = new Mat();
		Core.split(this.complexImage, this.planes);
		Core.normalize(this.planes.get(0), restoredImage, 0, 255, Core.NORM_MINMAX);
		
		// move back the Mat to 8 bit, in order to proper show the result
		restoredImage.convertTo(restoredImage, CvType.CV_8U);
		
		this.updateImageView(antitransformedImage, Utils.mat2Image(restoredImage));
		// set a fixed width
		this.antitransformedImage.setFitWidth(250);
		// preserve image ratio
		this.antitransformedImage.setPreserveRatio(true);
		
		// disable the button for performing the antitransformation
		this.antitransformButton.setDisable(true);
	}
	
	/**
	 * Optimize the image dimensions
	 * 
	 * @param image
	 *            the {@link Mat} to optimize
	 * @return the image whose dimensions have been optimized
	 */
	private Mat optimizeImageDim(Mat image)
	{
		// init
		Mat padded = new Mat();
		// get the optimal rows size for dft
		int addPixelRows = Core.getOptimalDFTSize(image.rows());
		// get the optimal cols size for dft
		int addPixelCols = Core.getOptimalDFTSize(image.cols());
		// apply the optimal cols and rows size to the image
		Core.copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(),
				Core.BORDER_CONSTANT, Scalar.all(0));
		
		return padded;
	}
	
	/**
	 * Optimize the magnitude of the complex image obtained from the DFT, to
	 * improve its visualization
	 * 
	 * @param complexImage
	 *            the complex image obtained from the DFT
	 * @return the optimized image
	 */
	private Mat createOptimizedMagnitude(Mat complexImage)
	{
		// init
		List<Mat> newPlanes = new ArrayList<>();
		Mat mag = new Mat();
		// split the comples image in two planes
		Core.split(complexImage, newPlanes);
		// compute the magnitude
		Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);
		
		// move to a logarithmic scale
		Core.add(Mat.ones(mag.size(), CvType.CV_32F), mag, mag);
		Core.log(mag, mag);
		// optionally reorder the 4 quadrants of the magnitude image
		this.shiftDFT(mag);
		// normalize the magnitude image for the visualization since both JavaFX
		// and OpenCV need images with value between 0 and 255
		// convert back to CV_8UC1
		mag.convertTo(mag, CvType.CV_8UC1);
		Core.normalize(mag, mag, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
		
		// you can also write on disk the resulting image...
		// Imgcodecs.imwrite("../magnitude.png", mag);
		
		return mag;
	}
	
	/**
	 * Reorder the 4 quadrants of the image representing the magnitude, after
	 * the DFT
	 * 
	 * @param image
	 *            the {@link Mat} object whose quadrants are to reorder
	 */
	private void shiftDFT(Mat image)
	{
		image = image.submat(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
		int cx = image.cols() / 2;
		int cy = image.rows() / 2;
		
		Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
		Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
		Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
		Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));
		
		Mat tmp = new Mat();
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);
		
		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}
	
	/**
	 * Set the current stage (needed for the FileChooser modal window)
	 * 
	 * @param stage
	 *            the stage
	 */
	public void setStage(Stage stage)
	{
		this.stage = stage;
	}
	
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
}
