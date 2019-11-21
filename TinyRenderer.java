
import java.awt.Graphics;
import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;

import java.util.ArrayList;
import java.util.Random;

import java.io.*;
import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;


public class TinyRenderer extends JPanel {
	public static final int width = 800;
	public static final int height = 800;

	public static BufferedImage image = null; // this image contains the rendered scene
	
	public static double vertices[] = null; // Point cloud. The size equals to the number of vertices*3. 
	//E.g: in order to access to the y component of vertex index i, you should write vertices[i*3+1]
	
	public static int triangles[] = null; // Collection of triangles. The size equals to the number of triangles*3. 
	//Each triplet references indices in the vertices[] array.
	
	 public static double[][] zbuffer = null;  // z-buffer array
	 
	 public static double[][] zbufferShadow =null;
	 double[][] M = null; //MShadow = ViewPort*Projection(ortho)*View(LookAt)*Model(Id) : Word to Screen Shadow buffer = zbufferShadow
	 //M is the transformation matrix from the object space to the shadow buffer screen space.

	 public static double uv[] = null; // Texture coordinates (u,v,0). The size equals to the number of vertices*3. 
		//E.g: in order to access to the v component of vertex index i, you should write vertices[i*3+1]
	 
	 public static int trianglesUV[] = null; // Triangles UV coords. The size equals to the number of triangles*3. 
		//Each triplet references indices in the uv[] array.
	 
	 public static double vertexnormals[] = null; // Vertex normal vectors in 3D. The size equals to the number of vertices*3.
	 //in order to access to the Z component of the normal vector of vertex index i, you should write vertexnormals[i*3+2] 
	
/**
 *  Constructor : initialises class attributes 
 */	
	public TinyRenderer(){
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		
	}

//************************************************************************************************************************************************	
	
/**
 * Loads only mesh vertices and triangles from a file 	
 * @param .obj file
 * @throws FileNotFoundException
 * @throws IOException
 */
	public void loadObjVerticesTriangles(String file) throws FileNotFoundException, IOException
	{
		File objFile = new File(file);
		FileReader fileReader = new FileReader(objFile);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		String line = null;
		ArrayList<Double> ALvertices = new ArrayList<Double>();
        ArrayList<Integer> ALtriangles = new ArrayList<Integer>();
		
		while (true) {
			line = bufferedReader.readLine();
			if (null == line) {
				break;
			}
			line = line.trim();
			String[] stringValues = line.split(" ");
			for (int i=0; i<stringValues.length; ++i ) {
				stringValues[i] = stringValues[i].trim();
			}

			if (line.length() == 0 || line.startsWith("#")) {
				continue;
			} else if (line.startsWith("v ")) {
				for (int i=1; i<stringValues.length; ++i ) {
					if (stringValues[i].length()==0) continue;
					ALvertices.add(Double.valueOf(stringValues[i]));
				}
			}  else if (line.startsWith("f ")) {
				for (int i=1; i<stringValues.length; ++i ) {
					if (stringValues[i].length()==0) continue;
					String[] tmp = stringValues[i].split("/");
					ALtriangles.add(Integer.valueOf(tmp[0])-1);
				}
			}
		}
		bufferedReader.close();
		
		vertices  =  new double[ALvertices.size()];
        triangles = new int[ALtriangles.size()];
        
        for(int i=0; i<ALvertices.size(); i++) //copy the vertices into their array
        	vertices[i] = ALvertices.get(i).doubleValue();
        
        for(int i=0; i<ALtriangles.size(); i++) //copy the triangles into their array
        	triangles[i] = ALtriangles.get(i).intValue();
       
        ALvertices = null; //free memory
        ALtriangles = null;
				
	}
	
//************************************************************************************************************************************************	
	/**
	 * Loads only mesh vertices and triangles from a file 	
	 * @param .obj file
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
		public void loadObjVertexNormals(String file) throws FileNotFoundException, IOException
		{
			File objFile = new File(file);
			FileReader fileReader = new FileReader(objFile);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			String line = null;
			ArrayList<Double> ALvertices = new ArrayList<Double>();
	       
			while (true) {
				line = bufferedReader.readLine();
				if (null == line) {
					break;
				}
				line = line.trim();
				String[] stringValues = line.split(" ");
				for (int i=0; i<stringValues.length; ++i ) {
					stringValues[i] = stringValues[i].trim();
				}

				if (line.length() == 0 || line.startsWith("#")) {
					continue;
				} else if (line.startsWith("vn ")) {
					for (int i=1; i<stringValues.length; ++i ) {
						if (stringValues[i].length()==0) continue;
						ALvertices.add(Double.valueOf(stringValues[i]));
					}
				}  
			}
			bufferedReader.close();
			
			vertexnormals  =  new double[ALvertices.size()];    
	        for(int i=0; i<ALvertices.size(); i++) //copy the normal components into their array
	        	vertexnormals[i] = ALvertices.get(i).doubleValue();
	        ALvertices = null; //free memory

					
		}
		
//************************************************************************************************************************************************	
	
/**
 * Bresenham's Line Drawing Algorithm between (x0,y0) and (x1,y1) 	
 * @param x0
 * @param y0
 * @param x1
 * @param y1
 * @param color
 */
	public void line(int x0, int y0, int x1, int y1, int color) {
		if (x0==x1 && y0==y1) return;
		boolean steep = false;
		if (Math.abs(y1-y0) > Math.abs(x1-x0)) {
			steep = true;
			int tmp1 = x0, tmp2 = x1;
			x0 = y0;
			x1 = y1;
			y0 = tmp1;
			y1 = tmp2;
		}
		if (x0>x1) {
			int tmp1 = x0, tmp2 = y0;
			x0 = x1;
			x1 = tmp1;
			y0 = y1;
			y1 = tmp2;
		}
		int dx = x1-x0;
		int dy = y1-y0;
		int derror2 = 2*Math.abs(dy);
		int error2 = 0;
		int y = y0;
		for (int x=x0; x<x1; x++) {
			if (steep) {
				if (y>=0 && y<width && x>=0 && x<height) {
					image.setRGB(y, x, color);
				}
			} else {
				if (x>=0 && x<width && y>=0 && y<height) {
					image.setRGB(x, y, color);
				}
			}
			error2 += derror2;
			if (error2>dx) {
				y += (y1>y0?1:-1);
				error2 -= 2*dx;
			}
		}
	}
	
/**
 * Draw the wire mesh 
 */
	public void drawWireMesh(){

		//int red   = new Color(255,   0,   0).getRGB();
		//int green = new Color(  0, 255,   0).getRGB();
		//int blue  = new Color(  0,   0, 255).getRGB();
		//int white = new Color(255, 255, 255).getRGB();
		int yellow = new Color(255, 255, 0).getRGB();


		for (int t=0; t<triangles.length/3; t++) {
			for (int e=0; e<3; e++) {
				Double x0 = vertices[triangles[t*3+e]*3+0];
				Double y0 = vertices[triangles[t*3+e]*3+1];
				Double x1 = vertices[triangles[t*3+(e+1)%3]*3+0];
				Double y1 = vertices[triangles[t*3+(e+1)%3]*3+1];

				int ix0 = (int)(width*(x0+1.f)/2.f+.5f);
				int ix1 = (int)(width*(x1+1.f)/2.f+.5f);
				int iy0 = (int)(height*(1.f-y0)/2.f+.5f);
				int iy1 = (int)(height*(1.f-y1)/2.f+.5f);
				line(ix0, iy0, ix1, iy1,yellow);
			}
		}		
	}

//*************************************************************************************************************************************************	
	
	
/**
 * Compute the matrix product A*B
 * @param A
 * @param B
 * @return
 */
	public double[][] matrix_product(double[][] A, double[][] B) {
		if (A.length==0 || A[0].length != B.length)
			throw new IllegalStateException("invalid dimensions");

		double[][] matrix = new double[A.length][B[0].length];
		for (int i=0; i<A.length; i++) {
			for (int j=0; j<B[0].length; j++) {
				double sum = 0;
				for (int k=0; k<A[i].length; k++)
					sum += A[i][k]*B[k][j];
				matrix[i][j] = sum;
			}
		}
		return matrix;
	}


/**
 * Invert the matrix.
 * N.B. it works for 3x3 matrices only!
 * @param m
 * @return
 */
	public double[][] matrix_inverse_BAD(double[][] m) {
		if (m[0].length != m.length || m.length != 3)
			throw new IllegalStateException("invalid dimensions");
		double[][] inverse = new double[m.length][m.length];
		double invdet = 1. / (m[0][0]*(m[1][1]*m[2][2] - m[2][1]*m[1][2]) - m[0][1]*(m[1][0]*m[2][2] 
				             - m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]));
		inverse[0][0] = (m[1][1]*m[2][2] - m[2][1]*m[1][2])*invdet;
		inverse[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2])*invdet;
		inverse[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1])*invdet;
		inverse[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2])*invdet;
		inverse[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0])*invdet;
		inverse[1][2] = (m[1][0]*m[0][2] - m[0][0]*m[1][2])*invdet;
		inverse[2][0] = (m[1][0]*m[2][1] - m[2][0]*m[1][1])*invdet;
		inverse[2][1] = (m[2][0]*m[0][1] - m[0][0]*m[2][1])*invdet;
		inverse[2][2] = (m[0][0]*m[1][1] - m[1][0]*m[0][1])*invdet;
		return inverse;
	}
	
	public  double[][] matrix_inverse(double[][] m) {
        if (m[0].length != m.length || m.length != 3)
            throw new IllegalStateException("invalid dimensions");
        double[][] inverse = new double[m.length][m.length];
        double det = m[0][0]*(m[1][1]*m[2][2] - m[2][1]*m[1][2]) - m[0][1]*(m[1][0]*m[2][2] 
        		            - m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
        if (Math.abs(det)<1e-6)
            throw new IllegalStateException("non-invertible matrix");
        inverse[0][0] = (m[1][1]*m[2][2] - m[2][1]*m[1][2])/det;
        inverse[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2])/det;
        inverse[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1])/det;
        inverse[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2])/det;
        inverse[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0])/det;
        inverse[1][2] = (m[1][0]*m[0][2] - m[0][0]*m[1][2])/det;
        inverse[2][0] = (m[1][0]*m[2][1] - m[2][0]*m[1][1])/det;
        inverse[2][1] = (m[2][0]*m[0][1] - m[0][0]*m[2][1])/det;
        inverse[2][2] = (m[0][0]*m[1][1] - m[1][0]*m[0][1])/det;
        return inverse;
    }

/**
 * Verify if the point (x,y) lies inside the triangle [(x0,y0), (x1,y1), (x2,y2)]
 * @param x0
 * @param y0
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 * @param x
 * @param y
 * @return
 */
	public boolean in_triangle(int x0, int y0, int x1, int y1, int x2, int y2, int x, int y) {
		double[][] A = { { x0, x1, x2 }, { y0, y1, y2 }, { 1., 1., 1. } };
		double[][] b = { { x }, { y }, { 1. } };
		double[][] coord = matrix_product(matrix_inverse(A), b);
		return coord[0][0]>=0 && coord[1][0]>=0 && coord[2][0]>=0;
	}

	
/**
 * Draw flat shaded triangles with random color	
 */
	public void drawTriangleMesh(){
		Random rand = new Random();
		//int color = new Color(255, 255, 255).getRGB();

        for (int t=0; t< triangles.length/3; t++) { // iterate through all triangles
            int color = new Color(rand.nextInt(256), rand.nextInt(256), rand.nextInt(256)).getRGB();
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3];
            for (int v=0; v<3; v++) {
                double xw = vertices[triangles[t*3+v]*3+0]; // world coordinates
                double yw = vertices[triangles[t*3+v]*3+1];
                x[v] = (int)( width*(xw+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw)/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
	            for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
	                for (int py=bbminy; py<=bbmaxy; py++) {
	                    if (!in_triangle(x[0], y[0], x[1], y[1], x[2], y[2], px, py)) continue;
	                    image.setRGB(px, py, color);
	                }
	            }
            }catch (IllegalStateException ex) {}
            
        }
	}
	
//*********************************************************************************************************************************************	

	
/**
* Transpose the matrix
* @param matrix
* @return
*/
	public  double[][] matrix_transpose(double[][] matrix) {
		double[][] transpose = new double[matrix[0].length][matrix.length];

		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[i].length; j++)
				transpose[j][i] = matrix[i][j];
		return transpose;
	}

/**
 * Given a triangle [(x0,y0), (x1,y1), (x2,y2)],
 * compute barycentric coordinates of the point (x,y) w.r.t the triangle
 * @param x0
 * @param y0
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 * @param x
 * @param y
 * @return
 */
    public double[] barycentric_coords(int x0, int y0, int x1, int y1, int x2, int y2, int x, int y) {
        double[][] A = { { x0, x1, x2 }, { y0, y1, y2 }, { 1., 1., 1. } };
        double[][] b = { { x }, { y }, { 1. } };
        return matrix_transpose(matrix_product(matrix_inverse(A), b))[0];
    }


/**
 * Draw flat shaded triangles with random color and hidden faces removal (z buffer)
 */	
public void drawTriangleMeshZBuffer(){
		
	zbuffer = new double[width][height]; // initialize the z-buffer
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            zbuffer[i][j] = -1.;
        }
    }
    
    Random rand = new Random();
    
    for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
    	int color = new Color(rand.nextInt(256), rand.nextInt(256), rand.nextInt(256)).getRGB(); // one color per triangle
        double[] xw = new double[3]; // triangle in world coordinates
        double[] yw = new double[3];
        double[] zw = new double[3];
        int[] x = new int[3]; // triangle in screen coordinates
        int[] y = new int[3];
        for (int v=0; v<3; v++) {
            xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
            yw[v] = vertices[triangles[t*3+v]*3+1];
            zw[v] = vertices[triangles[t*3+v]*3+2];
            x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
            y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
        }

        int bbminx = width-1; // screen bounding box for the triangle to rasterize
        int bbminy = height-1;
        int bbmaxx = 0;
        int bbmaxy = 0;
        for (int v=0; v<3; v++) {
            bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
            bbminy = Math.max(0, Math.min(bbminy, y[v]));
            bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
            bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
        }
        try { // non-ivertible matrix (can happen if a triangle is degenerate)
            for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                for (int py=bbminy; py<=bbmaxy; py++) {
                    double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                    if (coord[0]<0. || coord[1]<0. || coord[2]<0.) continue; // discard the point outside the triangle
                    double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                    if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                    zbuffer[px][py] = pz;
                    image.setRGB(px, py, color);
                }
            }
        } catch (IllegalStateException ex) {}
    }
    
}
	
	
//*******************************************************************************************************************************************

/**
 * Dot product between two vectors
 * N.B. works for dimension 3 vectors only
 * @param v1
 * @param v2
 * @return
 */
    public double dot_product(double[] v1, double[] v2) {
        if (v1.length != v2.length || v1.length != 3)
            throw new IllegalStateException("invalid dimensions");
        return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
    }
 
/**
 * Cross product between two vectors
 * N.B. works for dimension 3 vectors only
 * @param v1
 * @param v2
 * @return
 */
    public double[] cross_product(double[] v1, double[] v2) {
        if (v1.length != v2.length || v1.length != 3)
            throw new IllegalStateException("invalid dimensions");
        double[] cross = new double[3];
        cross[0] = v1[1]*v2[2] - v1[2]*v2[1];
        cross[1] = v1[2]*v2[0] - v1[0]*v2[2];
        cross[2] = v1[0]*v2[1] - v1[1]*v2[0];
        return cross;
    }

/**
 * Given a triangle, return its normal
 * N.B. works for dimension 3
 * @param x
 * @param y
 * @param z
 * @return
 */
    public double [] triangle_normal(double[] x, double[] y, double[] z) {
        if (x.length != y.length || x.length != z.length || x.length != 3)
            throw new IllegalStateException("invalid dimensions");
        double[] edge_a = {x[1] - x[0], y[1] - y[0], z[1] - z[0]};
        double[] edge_b = {x[2] - x[0], y[2] - y[0], z[2] - z[0]};
        double[] cross = cross_product(edge_a, edge_b);
        double norm = Math.sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
        if (norm<1e-6)
            throw new IllegalStateException("degenerate triangle");
        cross[0] /= norm;
        cross[1] /= norm;
        cross[2] /= norm;
        return cross;
    }
	
 //****************************************************************************************************************************	
/**
 * Flat (Lambert) shading.
 * One illumination calculation per triangle. 
 * Assign all pixels inside each triangle the same color (white).
 */
	public void FlatShading(){
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        double[] light = {1., 1., 1. };
        //double[] light = {1., 1., 0. };
    	double normL = Math.sqrt(light[0]*light[0] + light[1]*light[1] + light[2]*light[2]);
    	light[0] = light[0]/normL;
    	light[1] = light[1]/normL;
    	light[2] = light[2]/normL;
    	    	
    	 double[][] ViewPort = {
    	    		{width/2.,    0.,       0.,    width/2.},
    	    		{0.      ,  -height/2., 0,     height/2.},
    	    		{0.,          0.,       1.,    1.},
    	    		{0.,          0.,       0.,    1.}
    	    		
    	    };
        
        for (int t=0; t< triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3];
            
            double[] xS = new double[3]; // triangle in ViewPort coordinates
            double[] yS = new double[3];
            double[] zS = new double[3];
            
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                
                xS[v] = ViewPort[0][0]*xw[v] + ViewPort[0][1]*yw[v] + ViewPort[0][2]*zw[v] + ViewPort[0][3];
                yS[v] = ViewPort[1][0]*xw[v] + ViewPort[1][1]*yw[v] + ViewPort[1][2]*zw[v] + ViewPort[1][3];
                zS[v] = ViewPort[2][0]*xw[v] + ViewPort[2][1]*yw[v] + ViewPort[2][2]*zw[v] + ViewPort[2][3];
                double w      = ViewPort[3][0]*xw[v] + ViewPort[3][1]*yw[v] + ViewPort[3][2]*zw[v] + ViewPort[3][3];
                xS[v] = xS[v]/w;
                yS[v] = yS[v]/w;
                //zS[v] = zS[v]/w;
                
                
                // world-to-screen transformation
                //x[v] = (int)( width*(xw[v]/2 + 0.5));  // (xw+1)*width/2
               //x[v] = (int)( width/2*xw[v] + width/2);
                //y[v] = (int)(height*(-yw[v]/2 + 0.5)); // -[ (y+1)*height/2 ]
               //y[v] = (int)(height/2*(-yw[v]) + height/2); 
             // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
                
                x[v] = (int) xS[v];
                y[v] = (int) yS[v];
                
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
                        
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
                        
                        double[] normal = triangle_normal(xw, yw, zw); 
                      //int intensityDiffuse = (int)Math.min(255, Math.max(0, 255*dot_product(normal, new double[]{0.3, 0.3, 1.})));  
                    	// triangle intensity is the (clamped) cosine of the angle between the triangle normal and the light direction 
                        double diffuse = Math.max(0, dot_product(normal, light));
                    	int intensity = (int) Math.min(255, 255*diffuse) ; 
                    	int color = new Color(intensity, intensity, intensity).getRGB(); //color for the current TRIANGLE
   
                        image.setRGB(px, py, color);
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
//***************************************************************************************************************************************	
	
	public void saveZBuffer()
	{
		 BufferedImage zbufferImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB );
		 
		 double min = Integer.MAX_VALUE;
		 double max = Integer.MIN_VALUE;
		 
	        for (int i=0; i<width; i++) {
	            for (int j=0; j<height; j++) {
	            	
	            	 
	            	 if(zbuffer[i][j]<min) min = zbuffer[i][j];
	            	 if(zbuffer[i][j]>=max) max = zbuffer[i][j];
	            	
	            }
	        }
	        
	        double delta = max-min;
	        
	        for (int i=0; i<width; i++) {
	            for (int j=0; j<height; j++) {
	            	
	            	int intensity =(int) (((zbuffer[i][j] - min)) * (255/delta) );
            		//System.out.println(" "+  intensity);
            		int color = new Color(intensity , intensity, intensity)  .getRGB();
	            	zbufferImage.setRGB(i, j, color);
	            	
	            }
	        } 
	        
	        try {
				ImageIO.write(zbufferImage, "png", new File("zbuffer.png"));
			} catch (IOException ex) {ex.printStackTrace();}
	        
	        
	        System.out.println("ZBuffer MIN value = "+  min);  
	        System.out.println("ZBuffer MAX value = "+  max);
	        System.out.println("ZBuffer Delta = " +  delta);
	        
	        
	}

//************************************************************************************
	public void saveZShadowBuffer()
	{
		 BufferedImage zbufferImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB );
		 
		 double min = Integer.MAX_VALUE;
		 double max = Integer.MIN_VALUE;
		 
	        for (int i=0; i<width; i++) {
	            for (int j=0; j<height; j++) {
	            	
	            	 
	            	 if(zbufferShadow[i][j]<min) min = zbufferShadow[i][j];
	            	 if(zbufferShadow[i][j]>=max) max = zbufferShadow[i][j];
	            	
	            }
	        }
	        
	        double delta = max-min;
	        
	        for (int i=0; i<width; i++) {
	            for (int j=0; j<height; j++) {
	            	
	            	int intensity =(int) (((zbufferShadow[i][j] - min)) * (255/delta) );
            		//System.out.println(" "+  intensity);
            		int color = new Color(intensity , intensity, intensity)  .getRGB();
	            	zbufferImage.setRGB(i, j, color);
	            	
	            }
	        } 
	        
	        try {
				ImageIO.write(zbufferImage, "png", new File("zbufferShadow.png"));
			} catch (IOException ex) {ex.printStackTrace();}
	        
	        
	        System.out.println("ZBuffer MIN value = "+  min);  
	        System.out.println("ZBuffer MAX value = "+  max);
	        System.out.println("ZBuffer Delta = " +  delta);
	        
	        
	}

	
	
//****************************************************************************************************************************
	public void loadTextureCoords(String file) throws FileNotFoundException, IOException
	{
		File objFile = new File(file);
		FileReader fileReader = new FileReader(objFile);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		String line = null;
		
        ArrayList<Double> ALuv = new ArrayList<Double>();
        ArrayList<Integer> ALTriangleUVIndex = new ArrayList<Integer>();
		
        //int count =0;

		while (true) {
			line = bufferedReader.readLine();
			if (null == line) {
				break;
			}
			line = line.trim();
			String[] stringValues = line.split(" ");
			for (int i=0; i<stringValues.length; ++i ) {
				stringValues[i] = stringValues[i].trim();
			}

			if (line.length() == 0 || line.startsWith("#")) {
				continue;
			} else if (line.startsWith("vt ")) {
				for (int i=1; i<stringValues.length; ++i ) {
					if (stringValues[i].length()==0) continue;
					ALuv.add(Double.valueOf(stringValues[i]));
				}
			}else if (line.startsWith("f ")) {
				for (int i=1; i<stringValues.length; ++i ) {
					if (stringValues[i].length()==0) continue;
					String[] tmp = stringValues[i].split("/");
					ALTriangleUVIndex.add((Integer.valueOf(tmp[1])-1));
					//count++;
					//if(count == 2492*3)
						//System.out.println(" " + (Integer.valueOf(tmp[0])-1)+" " + (Integer.valueOf(tmp[1])-1)+" " + (Integer.valueOf(tmp[2])-1));
				}
			}
		}
		bufferedReader.close();
		
		System.out.println("ALuv.size() = "+ALuv.size());		
		uv  =  new double[ALuv.size()];
        for(int i=0; i<ALuv.size(); i++) //copy the texture coords into their array
        	uv[i] = ALuv.get(i).doubleValue();
        ALuv = null;
        
        System.out.println("ALTriangleUVIndex.size() = "+ALTriangleUVIndex.size());		
        trianglesUV=new int[ALTriangleUVIndex.size()];
        for(int i=0; i<ALTriangleUVIndex.size();i++)
        	trianglesUV[i]=ALTriangleUVIndex.get(i);
        ALTriangleUVIndex=null;
	}
//----------------------------------------------------------------------------------------------------------------
/**
 * Flat (Lambert) shading.
 * One illumination calculation per triangle. Assign all pixels inside each triangle the same color.
 * Uses texture colors of the triangle vertices and makes an average color.
 */
	public void FlatShading_First()
	{
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
		
		try {
			loadTextureCoords("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
//		BufferedImage texture = null;
//		int TexWidth=0; 
//		int TexHeight=0; 
//		try
//	    {
//	      // the line that reads the texture image file
//		  texture = ImageIO.read(new File("african_head_diffuse.png"));
//	      System.out.println("Texture : height = "+texture.getHeight() );
//	      System.out.println("Texture : width  = "+texture.getWidth() );
//	      TexWidth = texture.getWidth();
//	      TexHeight = texture.getHeight();
//	      // work with the image here ...
//	      //image = texture;
//	      
//	    } 
//	    catch (IOException e) { e.printStackTrace();}
//		
		
        double[] light = new double[]{1., 1., 1.};
        double normL = Math.sqrt(light[0]*light[0] + light[1]*light[1] + light[2]*light[2]);
    	light[0] = light[0]/normL;
    	light[1] = light[1]/normL;
    	light[2] = light[2]/normL;
		
		 for (int t=0; t< triangles.length/3  ; t++) { // iterate through all triangles
	            double[] xw = new double[3]; // triangle in world coordinates
	            double[] yw = new double[3];
	            double[] zw = new double[3];
	            int[] x = new int[3]; // triangle in screen coordinates
	            int[] y = new int[3];
//	            int[] TV = new int[3]; // triangle texture vertices
//	            double[] utex = new double[3]; // u tex coords
//	            double[] vtex = new double[3];
	            for (int v=0; v<3; v++) {
	                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
	                	//System.out.println("xw[v] = "+ xw[v]);
	                yw[v] = vertices[triangles[t*3+v]*3+1];
	                	//System.out.println("yw[v] = "+ yw[v]);
	                zw[v] = vertices[triangles[t*3+v]*3+2];
	                	//System.out.println("zw[v] = "+ zw[v]);
	                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
	                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
	                
//	                TV[v] = trianglesUV[t*3+v];
//	                	//System.out.println("TV[v] = "+ TV[v]);
//	                utex[v] = uv[trianglesUV[t*3+v]*3+0];
//	                	//System.out.println("utex[v] = " +utex[v]);
//	                vtex[v] = uv[trianglesUV[t*3+v]*3+1];
//	                	//System.out.println("vtex[v] = " +vtex[v]);
	            }
	            
//	            int UImg0 = (int)(utex[0]*TexWidth);
//	            int VImg0 = (int)((1-vtex[0])*TexHeight); // v flipped !
//	            	//System.out.println("Pixel0 at : "+ UImg0 + " , "+VImg0 );
//	            Color pixel0 = new Color( texture.getRGB( UImg0, VImg0 ) );
//	            //System.out.println("Color 1 = " + pixel0);
//	            
//	            int UImg1 = (int)(utex[1]*TexWidth);
//	            int VImg1 = (int)((1-vtex[1])*TexHeight); // v flipped !
//	            	//System.out.println("Pixel1 at : "+ UImg1 + " , "+VImg1 );
//	            Color pixel1 = new Color( texture.getRGB( UImg1, VImg1 ) );
//	            //System.out.println("Color 2 = " + pixel1);
//	            
//	            int UImg2 = (int)(utex[2]*TexWidth);
//	            int VImg2 = (int)((1-vtex[2])*TexHeight); // v flipped !
//	            	//System.out.println("Pixel2 at : "+ UImg2 + " , "+VImg2 );
//	            Color pixel2 = new Color( texture.getRGB( UImg2, VImg2 ) );
//	            	//System.out.println("Color 3 = " + pixel2);
//	            
//	            int rtmp = (pixel0.getRed()+ pixel1.getRed()+pixel2.getRed())/3;
//	            int gtmp = (pixel0.getGreen() + pixel1.getGreen() +pixel2.getGreen() )/3;
//	            int btmp = (pixel0.getBlue() + pixel1.getBlue() + pixel2.getBlue()  )/3; 
//	            Color COLOR = new Color(rtmp, gtmp,btmp); // average color !!!
//	            	//System.out.println("COLOR = " + COLOR);
//	            
//	            double[] normal = triangle_normal(xw, yw, zw);
//	            double dotProduct = dot_product(normal, new double[]{0., 0., 1.});
//                
//                int r = (int)Math.min(255, Math.max(0, COLOR.getRed()   *dotProduct )); 
//                int g = (int)Math.min(255, Math.max(0, COLOR.getGreen() *dotProduct )); 
//                int b = (int)Math.min(255, Math.max(0, COLOR.getBlue()  *dotProduct )); 
//              
//                Color color = new Color(r,g,b);
	            
	            int bbminx = width-1; // screen bounding box for the triangle to rasterize
	            int bbminy = height-1;
	            int bbmaxx = 0;
	            int bbmaxy = 0;
	            for (int v=0; v<3; v++) {
	                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
	                bbminy = Math.max(0, Math.min(bbminy, y[v]));
	                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
	                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
	            }
	            try { // non-ivertible matrix (can happen if a triangle is degenerate)
	                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
	                    for (int py=bbminy; py<=bbmaxy; py++) {
	                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
	                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle       
	                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
	                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
	                        zbuffer[px][py] = pz;
	                        
	                        double[] normal = triangle_normal(xw, yw, zw);
	        	            double dotProduct = dot_product(normal, light);
	                        
	                        int r = (int)Math.min(255, Math.max(0, 255   *dotProduct )); 
	                       
	                        Color color = new Color(r,r,r);
	                        
	                        image.setRGB(px, py, color.getRGB());
	                    }
	                }
	            } catch (IllegalStateException ex) {}
	        }//end for triangles
	}
	
//***************************************************************************************************************	
/**
 * UV coordinates interpolation only 	
 */
	
	public void UVInterpolateOnly()
	{
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
		
		try {
			loadTextureCoords("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		BufferedImage texture = null;
		int TexWidth=0; 
		int TexHeight=0; 
		try
	    {
	      // the line that reads the texture image file
		  texture = ImageIO.read(new File("african_head_diffuse.png"));
	      System.out.println("Texture : height = "+texture.getHeight() );
	      System.out.println("Texture : width  = "+texture.getWidth() );
	      TexWidth = texture.getWidth();
	      TexHeight = texture.getHeight();
	      // work with the image here ...
	      //image = texture;
	      
	    } 
	    catch (IOException e) { e.printStackTrace();}
		
		
		 for (int t=0; t< triangles.length/3  ; t++) { // iterate through all triangles
	            double[] xw = new double[3]; // triangle in world coordinates
	            double[] yw = new double[3];
	            double[] zw = new double[3];
	            int[] x = new int[3]; // triangle in screen coordinates
	            int[] y = new int[3];
	            int[] TV = new int[3]; // triangle texture vertices
	            double[] utex = new double[3]; // u tex coords
	            double[] vtex = new double[3];
	            for (int v=0; v<3; v++) {
	                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
	                	//System.out.println("xw[v] = "+ xw[v]);
	                yw[v] = vertices[triangles[t*3+v]*3+1];
	                	//System.out.println("yw[v] = "+ yw[v]);
	                zw[v] = vertices[triangles[t*3+v]*3+2];
	                	//System.out.println("zw[v] = "+ zw[v]);
	                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
	                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
	                
	                TV[v] = trianglesUV[t*3+v];
	                	//System.out.println("TV[v] = "+ TV[v]);
	                utex[v] = uv[trianglesUV[t*3+v]*3+0];
	                	//System.out.println("utex[v] = " +utex[v]);
	                vtex[v] = uv[trianglesUV[t*3+v]*3+1];
	                	//System.out.println("vtex[v] = " +vtex[v]);
	            }
	            
	        
	            int bbminx = width-1; // screen bounding box for the triangle to rasterize
	            int bbminy = height-1;
	            int bbmaxx = 0;
	            int bbmaxy = 0;
	            for (int v=0; v<3; v++) {
	                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
	                bbminy = Math.max(0, Math.min(bbminy, y[v]));
	                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
	                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
	            }
	            try { // non-ivertible matrix (can happen if a triangle is degenerate)
	                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
	                    for (int py=bbminy; py<=bbmaxy; py++) {
	                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
	                        if (coord[0]<0. || coord[1]<0. || coord[2]<0.) continue; // discard the point outside the triangle
	                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
	                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
	                        zbuffer[px][py] = pz;
	                    
	                        //UV coords interpolation  
	                        double currentU = (coord[0]*utex[0] + coord[1]*utex[1] + coord[2]*utex[2]);
	                        double currentV = (coord[0]*vtex[0] + coord[1]*vtex[1] + coord[2]*vtex[2]);                        
	                        int A = (int)(currentU*TexWidth);
	        	            int B = (int)((1-currentV)*TexHeight); // v flipped !
	        	            Color col = new Color( texture.getRGB( A, B ) );	                        
	                        image.setRGB(px, py, col.getRGB()); 
	                    }
	                }
	            } catch (IllegalStateException ex) {}
	        }//end for triangles
	}

//******************************************************************************************************************************************
	public void FlatShadingUVTextureInterpolation()
			{
				
				try {
					loadTextureCoords("african_head.obj");
				} 
				catch (FileNotFoundException e) {e.printStackTrace();} 
				catch (IOException e) {e.printStackTrace();}
				
				BufferedImage texture = null;
				int TexWidth=0; 
				int TexHeight=0; 
				try
			    {
			      // the line that reads the texture image file
				  texture = ImageIO.read(new File("african_head_diffuse.png"));
			      System.out.println("Texture : height = "+texture.getHeight() );
			      System.out.println("Texture : width  = "+texture.getWidth() );
			      TexWidth = texture.getWidth();
			      TexHeight = texture.getHeight();
			      // work with the image here ...
			      //image = texture;
			      
			    } 
			    catch (IOException e) { e.printStackTrace();}
				
				zbuffer = new double[width][height]; // initialize the z-buffer
		        for (int i=0; i<width; i++) {
		            for (int j=0; j<height; j++) {
		                zbuffer[i][j] = -1.;
		            }
		        }
		        
		        for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
		            double[] xw = new double[3]; // triangle in world coordinates
		            double[] yw = new double[3];
		            double[] zw = new double[3];
		            int[] x = new int[3]; // triangle in screen coordinates
		            int[] y = new int[3]; 
		            int[] TV = new int[3]; // triangle texture vertices
		            double[] utex = new double[3]; // u tex coords
		            double[] vtex = new double[3];
		            for (int v=0; v<3; v++) {
		                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
		                yw[v] = vertices[triangles[t*3+v]*3+1];
		                zw[v] = vertices[triangles[t*3+v]*3+2];
		                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
		                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
		                
		                
		                TV[v] = trianglesUV[t*3+v];
		            	//System.out.println("TV[v] = "+ TV[v]);
		                utex[v] = uv[trianglesUV[t*3+v]*3+0];
		            	//System.out.println("utex[v] = " +utex[v]);
		                vtex[v] = uv[trianglesUV[t*3+v]*3+1];
		            	//System.out.println("vtex[v] = " +vtex[v]);
		            }

		            int bbminx = width-1; // screen bounding box for the triangle to rasterize
		            int bbminy = height-1;
		            int bbmaxx = 0;
		            int bbmaxy = 0;
		            for (int v=0; v<3; v++) {
		                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
		                bbminy = Math.max(0, Math.min(bbminy, y[v]));
		                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
		                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
		            }
		            
		            
		            double[] normal = triangle_normal(xw, yw, zw);
		            double dotProduct = dot_product(normal, new double[]{0., 0., 1.});
		            
		            try { // non-ivertible matrix (can happen if a triangle is degenerate)
		                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
		                    for (int py=bbminy; py<=bbmaxy; py++) {
		                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
		                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
		                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
		                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
		                        zbuffer[px][py] = pz;                      
		                        
		                      //UV coords interpolation  
		                        double currentU = (coord[0]*utex[0] + coord[1]*utex[1] + coord[2]*utex[2]);
		                        double currentV = (coord[0]*vtex[0] + coord[1]*vtex[1] + coord[2]*vtex[2]);                        
		                        int A = (int)(currentU*TexWidth);
		        	            int B = (int)((1-currentV)*TexHeight); // v flipped !
		        	            Color col = new Color( texture.getRGB( A, B ) );
		        	            
		        	            int r = (int)Math.min(255, Math.max(0, col.getRed()   *dotProduct )); 
		                        int g = (int)Math.min(255, Math.max(0, col.getGreen() *dotProduct )); 
		                        int b = (int)Math.min(255, Math.max(0, col.getBlue()  *dotProduct )); 	                      
		                        Color color = new Color(r,g,b);
		        	            
		                        image.setRGB(px, py, color.getRGB()); 
		                        
		                        
		                    }
		                }
		            } catch (IllegalStateException ex) {}
		        }//end for triangles
		        
			}
			
//******************************************************************************************************************************************
/**
 * Gouraud shading. Uses vertex normals to compute a color at each triangle vertex and interpolates these colors over the triangle.
 *  1. Load the normal at each polygon vertex.
    2. Apply an illumination model to each vertex to calculate the light intensity from the vertex normal.
    3. Interpolate the vertex intensities using bilinear interpolation over the surface triangle. 
    
    Gouraud shading is most often used to achieve continuous lighting on triangle surfaces 
    by computing the lighting at the corners of each triangle and linearly interpolating the resulting colours for each pixel covered by the triangle. 
    
    Gourard shading calculates the shade at each TRIANGLE vertex, and interpolates (smoothes) these shades across the triangle.
    
 */
	public void GouraudShading()
	{
		try {
			loadObjVertexNormals("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3]; 
            double[] xN = new double[3]; // x compponent of the vertex normals 
            double[] yN = new double[3]; // y 
            double[] zN = new double[3]; //z
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
                xN[v] = vertexnormals[triangles[t*3+v]*3+0]; // vertex normals
                yN[v] = vertexnormals[triangles[t*3+v]*3+1];
                zN[v] = vertexnormals[triangles[t*3+v]*3+2];
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
            
            double[] light = {0., 0., 1.};
            double[] normal0 = {xN[0], yN[0],zN[0]};
            //int intensity0 = (int)Math.min(255, Math.max(0, 255*dot_product(normal0, new double[]{0., 0., 1.})));
            double intensity0 = Math.max(0, dot_product(normal0, light));
            
            double[] normal1 = {xN[1], yN[1],zN[1]};
            //int intensity1 = (int)Math.min(255, Math.max(0, 255*dot_product(normal1, new double[]{0., 0., 1.})));
            double intensity1 = Math.max(0, dot_product(normal1, light));
            
            double[] normal2 = {xN[2], yN[2],zN[2]};
            //int intensity2 = (int)Math.min(255, Math.max(0, 255*dot_product(normal2, new double[]{0., 0., 1.})));
            double intensity2 = Math.max(0, dot_product(normal2, light));
            
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
                        
                        double intensitySmooth = (coord[0]*intensity0 + coord[1]*intensity1 +coord[2]*intensity2);  //interpolates the colors at the 3 vertices 
                        //intesitySmooth can be negatif !
                        intensitySmooth = Math.min(255, Math.max(0,255*intensitySmooth)); 
                        //System.out.println(intensitySmooth);
                        int color = new Color((int)intensitySmooth, (int)intensitySmooth, (int)intensitySmooth).getRGB();                    
                        image.setRGB(px, py, color);
                        
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
	
//******************************************************************************************************************************************
/**
 * * Gouraud shading. Uses vertex normals to compute a color at each triangle vertex and interpolates these colors over the triangle.
 *  1. Load the normal at each polygon vertex.
    2. Apply an illumination model to each vertex to calculate the light intensity from the vertex normal.
    3. Interpolate the vertex intensities using bilinear interpolation over the surface polygon. 
 */
	public void GouraudShadingTexture()
	{
		try {
			loadObjVertexNormals("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		try {
			loadTextureCoords("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		BufferedImage texture = null;
		int TexWidth=0; 
		int TexHeight=0; 
		try
	    {
	      // the line that reads the texture image file
		  texture = ImageIO.read(new File("african_head_diffuse.png"));
	      System.out.println("Texture : height = "+texture.getHeight() );
	      System.out.println("Texture : width  = "+texture.getWidth() );
	      TexWidth = texture.getWidth();
	      TexHeight = texture.getHeight();
	      // work with the image here ...
	      //image = texture;
	      
	    } 
	    catch (IOException e) { e.printStackTrace();}
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3]; 
            double[] xN = new double[3]; // x compponent of the vertex normals 
            double[] yN = new double[3]; // y 
            double[] zN = new double[3]; //z
            int[] TV = new int[3]; // triangle texture vertices
            double[] utex = new double[3]; // u tex coords
            double[] vtex = new double[3];
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
                
                xN[v] = vertexnormals[triangles[t*3+v]*3+0]; // vertex normals
                yN[v] = vertexnormals[triangles[t*3+v]*3+1];
                zN[v] = vertexnormals[triangles[t*3+v]*3+2];
                
                TV[v] = trianglesUV[t*3+v];
            	//System.out.println("TV[v] = "+ TV[v]);
                utex[v] = uv[trianglesUV[t*3+v]*3+0];
            	//System.out.println("utex[v] = " +utex[v]);
                vtex[v] = uv[trianglesUV[t*3+v]*3+1];
            	//System.out.println("vtex[v] = " +vtex[v]);
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
            
            int UImg0 = (int)(utex[0]*TexWidth);
            int VImg0 = (int)((1-vtex[0])*TexHeight); // v flipped !
            	//System.out.println("Pixel0 at : "+ UImg0 + " , "+VImg0 );
            Color pixel0 = new Color( texture.getRGB( UImg0, VImg0 ) );
            	//System.out.println("Color 1 = " + pixel0);
            
            int UImg1 = (int)(utex[1]*TexWidth);
            int VImg1 = (int)((1-vtex[1])*TexHeight); // v flipped !
            	//System.out.println("Pixel1 at : "+ UImg1 + " , "+VImg1 );
            Color pixel1 = new Color( texture.getRGB( UImg1, VImg1 ) );
            	//System.out.println("Color 2 = " + pixel1);
            
            int UImg2 = (int)(utex[2]*TexWidth);
            int VImg2 = (int)((1-vtex[2])*TexHeight); // v flipped !
            	//System.out.println("Pixel2 at : "+ UImg2 + " , "+VImg2 );
            Color pixel2 = new Color( texture.getRGB( UImg2, VImg2 ) );
            	//System.out.println("Color 3 = " + pixel2);
            
            double[] normal0 = {xN[0], yN[0],zN[0]};
            int intensity0RED   = (int)Math.min(255, Math.max(0, pixel0.getRed()  *dot_product(normal0, new double[]{0., 0., 1.})));
            int intensity0GREEN = (int)Math.min(255, Math.max(0, pixel0.getGreen()*dot_product(normal0, new double[]{0., 0., 1.})));
            int intensity0BLUE  = (int)Math.min(255, Math.max(0, pixel0.getBlue() *dot_product(normal0, new double[]{0., 0., 1.})));
            
            double[] normal1 = {xN[1], yN[1],zN[1]};
            int intensity1RED   = (int)Math.min(255, Math.max(0, pixel1.getRed()  *dot_product(normal1, new double[]{0., 0., 1.})));
            int intensity1GREEN = (int)Math.min(255, Math.max(0, pixel1.getGreen()*dot_product(normal1, new double[]{0., 0., 1.})));
            int intensity1BLUE  = (int)Math.min(255, Math.max(0, pixel1.getBlue() *dot_product(normal1, new double[]{0., 0., 1.})));
            
            double[] normal2 = {xN[2], yN[2],zN[2]};
            int intensity2RED   = (int)Math.min(255, Math.max(0, pixel2.getRed()  *dot_product(normal2, new double[]{0., 0., 1.})));
            int intensity2GREEN = (int)Math.min(255, Math.max(0, pixel2.getGreen()*dot_product(normal2, new double[]{0., 0., 1.})));
            int intensity2BLUE  = (int)Math.min(255, Math.max(0, pixel2.getBlue() *dot_product(normal2, new double[]{0., 0., 1.})));
            
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<0. || coord[1]<0. || coord[2]<0.) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
                        
                        int intensitySmoothRED =   (int)(coord[0]*intensity0RED   + coord[1]*intensity1RED   +coord[2]*intensity2RED);  
                        int intensitySmoothGREEN = (int)(coord[0]*intensity0GREEN + coord[1]*intensity1GREEN +coord[2]*intensity2GREEN);  
                        int intensitySmoothBLUE =  (int)(coord[0]*intensity0BLUE  + coord[1]*intensity1BLUE  +coord[2]*intensity2BLUE);  
                        int color = new Color(intensitySmoothRED, intensitySmoothGREEN, intensitySmoothBLUE).getRGB();                    
                        image.setRGB(px, py, color);
                        
                        
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
	
//***********************************************************************************************************

	public void GouraudShadingUVTextureInterpolation()
	{
		try {
			loadObjVertexNormals("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		try {
			loadTextureCoords("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		BufferedImage texture = null;
		int TexWidth=0; 
		int TexHeight=0; 
		try
	    {
	      // the line that reads the texture image file
		  texture = ImageIO.read(new File("african_head_diffuse.png"));
	      System.out.println("Texture : height = "+texture.getHeight() );
	      System.out.println("Texture : width  = "+texture.getWidth() );
	      TexWidth = texture.getWidth();
	      TexHeight = texture.getHeight();
	      // work with the image here ...
	      //image = texture;
	      
	    } 
	    catch (IOException e) { e.printStackTrace();}
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3]; 
            double[] xN = new double[3]; // x compponent of the vertex normals 
            double[] yN = new double[3]; // y 
            double[] zN = new double[3]; //z
            int[] TV = new int[3]; // triangle texture vertices
            double[] utex = new double[3]; // u tex coords
            double[] vtex = new double[3];
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
                
                xN[v] = vertexnormals[triangles[t*3+v]*3+0]; // vertex normals
                yN[v] = vertexnormals[triangles[t*3+v]*3+1];
                zN[v] = vertexnormals[triangles[t*3+v]*3+2];
                
                TV[v] = trianglesUV[t*3+v];
            	//System.out.println("TV[v] = "+ TV[v]);
                utex[v] = uv[trianglesUV[t*3+v]*3+0];
            	//System.out.println("utex[v] = " +utex[v]);
                vtex[v] = uv[trianglesUV[t*3+v]*3+1];
            	//System.out.println("vtex[v] = " +vtex[v]);
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
            
            
            double[] light = {0., 0., 1.};
            double[] normal0 = {xN[0], yN[0],zN[0]};
            double intensity0 = Math.max(0, dot_product(normal0, light));
            
            double[] normal1 = {xN[1], yN[1],zN[1]};
            double intensity1 = Math.max(0, dot_product(normal1, light));
            
            double[] normal2 = {xN[2], yN[2],zN[2]};
            double intensity2 = Math.max(0, dot_product(normal2, light));
            
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
                        
                        
                      //UV coords interpolation  
                        double currentU = (coord[0]*utex[0] + coord[1]*utex[1] + coord[2]*utex[2]);
                        double currentV = (coord[0]*vtex[0] + coord[1]*vtex[1] + coord[2]*vtex[2]);                        
                        int A = (int)(currentU*TexWidth);
        	            int B = (int)((1-currentV)*TexHeight); // v flipped !
        	            Color col = new Color( texture.getRGB( A, B ) ); // color from the texture 
        	                    	            
        	            double intensitySmooth = (coord[0]*intensity0 + coord[1]*intensity1 +coord[2]*intensity2);  //interpolates the light intesities at the 3 vertices 
                        //intesitySmooth can be negatif !
        	            int intensitySmoothRED   = (int)Math.min(255, Math.max(0,col.getRed()   *intensitySmooth));  // put a color
        	            int intensitySmoothGREEN = (int)Math.min(255, Math.max(0,col.getGreen() *intensitySmooth)); 
        	            int intensitySmoothBLUE  = (int)Math.min(255, Math.max(0,col.getBlue()  *intensitySmooth)); 
                        
                        int color = new Color(intensitySmoothRED, intensitySmoothGREEN, intensitySmoothBLUE).getRGB();                    
                        image.setRGB(px, py, color);
        	            
                        
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
	

//******************************************************************************************************************************************
/**
 * Phong shading is similar to Gouraud shading, except that instead of interpolating the light intensities, 
 * the normals are interpolated between the vertices. Thus, the specular highlights are computed much more precisely than in the Gouraud shading model:

    1. Load the normal at each polygon vertex.
    2. From bilinear interpolation compute a normal, Ni, for each pixel. (This must be renormalized each time.)
    3. Apply an illumination model to each pixel to calculate the light intensity from Ni.
    
    Phong shading calculates the normal at each vertex, and interpolates these normals across the surfaces.
    The light, and therefore shade, at *each* pixel is *individually* calculated from its unique surface normal
    
 */
	public void PhongShading()
	{
		try {
			loadObjVertexNormals("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3]; 
            double[] xN = new double[3]; // x compponent of the vertex normals 
            double[] yN = new double[3]; // y 
            double[] zN = new double[3]; //z
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
                xN[v] = vertexnormals[triangles[t*3+v]*3+0]; // vertex normals
                yN[v] = vertexnormals[triangles[t*3+v]*3+1];
                zN[v] = vertexnormals[triangles[t*3+v]*3+2];
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
            
            double[] normal0 = {xN[0], yN[0],zN[0]}; // normal at vertex 0
            double[] normal1 = {xN[1], yN[1],zN[1]}; // normal at vertex 1     
            double[] normal2 = {xN[2], yN[2],zN[2]}; // normal at vertex 2
            double[] light   = {0., 0., 1.}; 
            
            
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
                        
                        double[] NPix = new double[3]; // interpolated normal ot the current pixel
                        NPix[0] = coord[0]*normal0[0] + coord[1]*normal1[0] + coord[2]*normal2[0] ;
                        NPix[1] = coord[0]*normal0[1] + coord[1]*normal1[1] + coord[2]*normal2[1] ;
                        NPix[2] = coord[0]*normal0[2] + coord[1]*normal1[2] + coord[2]*normal2[2] ;               
                        double norm = Math.sqrt(NPix[0]*NPix[0] + NPix[1]*NPix[1] +  NPix[2]*NPix[2]);                       
                        NPix[0] = NPix[0]/norm;
                        NPix[1] = NPix[1]/norm;
                        NPix[2] = NPix[2]/norm;
                        
                        int intensity = (int)Math.min(255, Math.max(0, 255*dot_product(NPix, light))); 
                        // triangle intensity is the (clamped) cosine of the angle between the PIXEL normal and the light direction
                        int color = new Color(intensity, intensity, intensity).getRGB();  // color for the current PIXEL                  
                        image.setRGB(px, py, color);
                        
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
	
//******************************************************************************************************************************************
	public void PhongShadingUVTextureInterpolation()
	{
		try {
			loadObjVertexNormals("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		try {
			loadTextureCoords("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		BufferedImage texture = null;
		int TexWidth=0; 
		int TexHeight=0; 
		try
	    {
	      // the line that reads the texture image file
		  texture = ImageIO.read(new File("african_head_diffuse.png"));
	      System.out.println("Texture : height = "+texture.getHeight() );
	      System.out.println("Texture : width  = "+texture.getWidth() );
	      TexWidth = texture.getWidth();
	      TexHeight = texture.getHeight();
	      // work with the image here ...
	      //image = texture;
	      
	    } 
	    catch (IOException e) { e.printStackTrace();}
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3]; 
            double[] xN = new double[3]; // x compponent of the vertex normals 
            double[] yN = new double[3]; // y 
            double[] zN = new double[3]; //z
            int[] TV = new int[3]; // triangle texture vertices
            double[] utex = new double[3]; // u tex coords
            double[] vtex = new double[3];
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
                
                xN[v] = vertexnormals[triangles[t*3+v]*3+0]; // vertex normals
                yN[v] = vertexnormals[triangles[t*3+v]*3+1];
                zN[v] = vertexnormals[triangles[t*3+v]*3+2];
                
                TV[v] = trianglesUV[t*3+v];
            	//System.out.println("TV[v] = "+ TV[v]);
                utex[v] = uv[trianglesUV[t*3+v]*3+0];
            	//System.out.println("utex[v] = " +utex[v]);
                vtex[v] = uv[trianglesUV[t*3+v]*3+1];
            	//System.out.println("vtex[v] = " +vtex[v]);
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
            
            double[] normal0 = {xN[0], yN[0],zN[0]}; // normal at vertex 0 
            double[] normal1 = {xN[1], yN[1],zN[1]}; // normal at vertex 1
            double[] normal2 = {xN[2], yN[2],zN[2]}; // normal at vertex 2
            double[] light   = {0., 0., 1.}; // light vector
            
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
                        
                        
                      //UV coords interpolation  
                        double currentU = (coord[0]*utex[0] + coord[1]*utex[1] + coord[2]*utex[2]);
                        double currentV = (coord[0]*vtex[0] + coord[1]*vtex[1] + coord[2]*vtex[2]);                        
                        int A = (int)(currentU*TexWidth);
        	            int B = (int)((1-currentV)*TexHeight); // v flipped !
        	            Color col = new Color( texture.getRGB( A, B ) ); // color from the texture 
        	            
        	            double[] NPix = new double[3]; // normal ot the current pixel
                        NPix[0] = coord[0]*normal0[0] + coord[1]*normal1[0] + coord[2]*normal2[0] ;
                        NPix[1] = coord[0]*normal0[1] + coord[1]*normal1[1] + coord[2]*normal2[1] ;
                        NPix[2] = coord[0]*normal0[2] + coord[1]*normal1[2] + coord[2]*normal2[2] ;                  
                        double norm = Math.sqrt(NPix[0]*NPix[0] + NPix[1]*NPix[1] +  NPix[2]*NPix[2]);                       
                        NPix[0] = NPix[0]/norm;
                        NPix[1] = NPix[1]/norm;
                        NPix[2] = NPix[2]/norm;
                        
                        int intensityRED =   (int)Math.min(255, Math.max(0, col.getRed()  *dot_product(NPix, light))); 
                        int intensityGREEN = (int)Math.min(255, Math.max(0, col.getGreen()*dot_product(NPix, light))); 
                        int intensityBLUE =  (int)Math.min(255, Math.max(0, col.getBlue() *dot_product(NPix, light))); 
        	                
                        int color = new Color(intensityRED, intensityGREEN, intensityBLUE).getRGB(); 
                        image.setRGB(px, py, color);
                        
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
	

//******************************************************************************************************************************************
	public void PhongShadingTexture()
	{
		try {
			loadObjVertexNormals("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		try {
			loadTextureCoords("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		BufferedImage texture = null;
		int TexWidth=0; 
		int TexHeight=0; 
		try
	    {
	      // the line that reads the texture image file
		  texture = ImageIO.read(new File("african_head_diffuse.png"));
	      System.out.println("Texture : height = "+texture.getHeight() );
	      System.out.println("Texture : width  = "+texture.getWidth() );
	      TexWidth = texture.getWidth();
	      TexHeight = texture.getHeight();
	      // work with the image here ...
	      //image = texture;
	      
	    } 
	    catch (IOException e) { e.printStackTrace();}
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3]; 
            double[] xN = new double[3]; // x compponent of the vertex normals 
            double[] yN = new double[3]; // y 
            double[] zN = new double[3]; //z
            int[] TV = new int[3]; // triangle texture vertices
            double[] utex = new double[3]; // u tex coords
            double[] vtex = new double[3];
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
                
                xN[v] = vertexnormals[triangles[t*3+v]*3+0]; // vertex normals
                yN[v] = vertexnormals[triangles[t*3+v]*3+1];
                zN[v] = vertexnormals[triangles[t*3+v]*3+2];
                
                TV[v] = trianglesUV[t*3+v];
            	//System.out.println("TV[v] = "+ TV[v]);
                utex[v] = uv[trianglesUV[t*3+v]*3+0];
            	//System.out.println("utex[v] = " +utex[v]);
                vtex[v] = uv[trianglesUV[t*3+v]*3+1];
            	//System.out.println("vtex[v] = " +vtex[v]);
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
            
            int UImg0 = (int)(utex[0]*TexWidth);
            int VImg0 = (int)((1-vtex[0])*TexHeight); // v flipped !
            	//System.out.println("Pixel0 at : "+ UImg0 + " , "+VImg0 );
            Color pixel0 = new Color( texture.getRGB( UImg0, VImg0 ) );
            	//System.out.println("Color 1 = " + pixel0);
            
            int UImg1 = (int)(utex[1]*TexWidth);
            int VImg1 = (int)((1-vtex[1])*TexHeight); // v flipped !
            	//System.out.println("Pixel1 at : "+ UImg1 + " , "+VImg1 );
            Color pixel1 = new Color( texture.getRGB( UImg1, VImg1 ) );
            	//System.out.println("Color 2 = " + pixel1);
            
            int UImg2 = (int)(utex[2]*TexWidth);
            int VImg2 = (int)((1-vtex[2])*TexHeight); // v flipped !
            	//System.out.println("Pixel2 at : "+ UImg2 + " , "+VImg2 );
            Color pixel2 = new Color( texture.getRGB( UImg2, VImg2 ) );
            	//System.out.println("Color 3 = " + pixel2);
            
            double[] normal0 = {xN[0], yN[0],zN[0]};
            int intensity0RED   = (int)Math.min(255, Math.max(0, pixel0.getRed()  *dot_product(normal0, new double[]{0., 0., 1.})));
            int intensity0GREEN = (int)Math.min(255, Math.max(0, pixel0.getGreen()*dot_product(normal0, new double[]{0., 0., 1.})));
            int intensity0BLUE  = (int)Math.min(255, Math.max(0, pixel0.getBlue() *dot_product(normal0, new double[]{0., 0., 1.})));
            
            double[] normal1 = {xN[1], yN[1],zN[1]};
            int intensity1RED   = (int)Math.min(255, Math.max(0, pixel1.getRed()  *dot_product(normal1, new double[]{0., 0., 1.})));
            int intensity1GREEN = (int)Math.min(255, Math.max(0, pixel1.getGreen()*dot_product(normal1, new double[]{0., 0., 1.})));
            int intensity1BLUE  = (int)Math.min(255, Math.max(0, pixel1.getBlue() *dot_product(normal1, new double[]{0., 0., 1.})));
            
            double[] normal2 = {xN[2], yN[2],zN[2]};
            int intensity2RED   = (int)Math.min(255, Math.max(0, pixel2.getRed()  *dot_product(normal2, new double[]{0., 0., 1.})));
            int intensity2GREEN = (int)Math.min(255, Math.max(0, pixel2.getGreen()*dot_product(normal2, new double[]{0., 0., 1.})));
            int intensity2BLUE  = (int)Math.min(255, Math.max(0, pixel2.getBlue() *dot_product(normal2, new double[]{0., 0., 1.})));
            
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<0. || coord[1]<0. || coord[2]<0.) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
                        
                        double[] NPix = new double[3]; // normal ot the current pixel
                        NPix[0] = coord[0]*normal0[0] + coord[1]*normal1[0] + coord[2]*normal2[0] ;
                        NPix[1] = coord[0]*normal0[1] + coord[1]*normal1[1] + coord[2]*normal2[1] ;
                        NPix[2] = coord[0]*normal0[2] + coord[1]*normal1[2] + coord[2]*normal2[2] ;
                    
                        double norm = Math.sqrt(NPix[0]*NPix[0] + NPix[1]*NPix[1] +  NPix[2]*NPix[2]); 
                        
                        NPix[0] = NPix[0]/norm;
                        NPix[1] = NPix[1]/norm;
                        NPix[2] = NPix[2]/norm;

                        int intensitySmoothRED =   (int)(coord[0]*intensity0RED   + coord[1]*intensity1RED   +coord[2]*intensity2RED);  
                        int intensitySmoothGREEN = (int)(coord[0]*intensity0GREEN + coord[1]*intensity1GREEN +coord[2]*intensity2GREEN);  
                        int intensitySmoothBLUE =  (int)(coord[0]*intensity0BLUE  + coord[1]*intensity1BLUE  +coord[2]*intensity2BLUE);  
                        //int color = new Color(intensitySmoothRED, intensitySmoothGREEN, intensitySmoothBLUE).getRGB();    
                        
                        int intensityRED   = (int)Math.min(255,  Math.max(0, intensitySmoothRED   *dot_product(NPix, new double[]{0., 0., 1.}))); 
                        int intensityGREEN = (int)Math.min(255,  Math.max(0, intensitySmoothGREEN *dot_product(NPix, new double[]{0., 0., 1.}))); 
                        int intensityBLUE  = (int)Math.min(255,  Math.max(0, intensitySmoothBLUE  *dot_product(NPix, new double[]{0., 0., 1.}))); 
                        
                        int color = new Color(intensityRED, intensityGREEN, intensityBLUE).getRGB(); 
                        image.setRGB(px, py, color);
                        
                        
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
	
//******************************************************************************************************************************************
/**
 * Normal Mapping with or without texture 	
 */
	public void NormalMapping()
	{		
		try {
			loadTextureCoords("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
		
		BufferedImage texture = null;
		int TexWidth=0; 
		int TexHeight=0; 
		try
	    {
	      // the line that reads the texture image file
		  texture = ImageIO.read(new File("african_head_diffuse.png"));
	      System.out.println("Texture : height = "+texture.getHeight() );
	      System.out.println("Texture : width  = "+texture.getWidth() );
	      TexWidth = texture.getWidth();
	      TexHeight = texture.getHeight();
	      // work with the image here ...
	      //image = texture;
	      
	    } 
	    catch (IOException e) { e.printStackTrace();}
		
		//african_head_nm.png
		BufferedImage normalMap = null;
		int NormalMapWidth=0; 
		int NormalMapHeight=0; 
		try
	    {
	      // the line that reads the texture image file
		  normalMap = ImageIO.read(new File("african_head_nm.png"));
	      System.out.println("Texture : height = "+normalMap.getHeight() );
	      System.out.println("Texture : width  = "+normalMap.getWidth() );
	      NormalMapWidth  = normalMap.getWidth();
	      NormalMapHeight = normalMap.getHeight();
	      // work with the image here ...
	      //image = texture;
	      
	    } 
	    catch (IOException e) { e.printStackTrace();}
		
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3]; 
            int[] TV = new int[3]; // triangle texture vertices
            double[] utex = new double[3]; // u tex coords
            double[] vtex = new double[3];
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
                
                TV[v] = trianglesUV[t*3+v];
                utex[v] = uv[trianglesUV[t*3+v]*3+0];
                vtex[v] = uv[trianglesUV[t*3+v]*3+1];
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
             
            double[] light = {0., 0., 1.};
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
                        
                        
                      //UV coords interpolation  
                        double currentU = (coord[0]*utex[0] + coord[1]*utex[1] + coord[2]*utex[2]);
                        double currentV = (coord[0]*vtex[0] + coord[1]*vtex[1] + coord[2]*vtex[2]);                        
                        int A = (int)(currentU*TexWidth);
        	            int B = (int)((1-currentV)*TexHeight); // v flipped !
        	            Color col = new Color( texture.getRGB( A, B ) ); // color from the texture 
        	            
        	            double[] NPix = new double[3]; // normal ot the current pixel
        	            Color colNormalMap = new Color( normalMap.getRGB( A, B ) ); // color from the texture 
        	            NPix[0] = colNormalMap.getRed()  -128;
        	            NPix[1] = colNormalMap.getGreen()-128;
        	            NPix[2] = colNormalMap.getBlue() -128;       	            
        	            double norm = Math.sqrt(NPix[0]*NPix[0] + NPix[1]*NPix[1] +  NPix[2]*NPix[2]); 
                        NPix[0] = NPix[0]/norm;
                        NPix[1] = NPix[1]/norm;
                        NPix[2] = NPix[2]/norm;
        	            
                        
//                        int intensityRED =   (int)Math.min(255, Math.max(0, col.getRed()   *dot_product(NPix, light))); 
//                        int intensityGREEN = (int)Math.min(255, Math.max(0, col.getGreen() *dot_product(NPix, light))); 
//                        int intensityBLUE =  (int)Math.min(255, Math.max(0, col.getBlue()  *dot_product(NPix, light))); 
                        
                        int intensityRED =   (int)Math.min(255, Math.max(0, 255  *dot_product(NPix, light))); 
                        int intensityGREEN = (int)Math.min(255, Math.max(0, 255  *dot_product(NPix, light))); 
                        int intensityBLUE =  (int)Math.min(255, Math.max(0, 255  *dot_product(NPix, light))); 
        	                
                        int color = new Color(intensityRED, intensityGREEN, intensityBLUE).getRGB(); 
                        image.setRGB(px, py, color);
                        
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
	
//*****************************************************************************************************************************************
//******************************************************************************************************************************************	
public void SpecularShadingFlat(){
		
		zbuffer = new double[width][height]; // initialize the z-buffer
        for (int i=0; i<width; i++) {
            for (int j=0; j<height; j++) {
                zbuffer[i][j] = -1.;
            }
        }
        
        for (int t=0; t< triangles.length/3; t++) { // iterate through all triangles
            double[] xw = new double[3]; // triangle in world coordinates
            double[] yw = new double[3];
            double[] zw = new double[3];
            int[] x = new int[3]; // triangle in screen coordinates
            int[] y = new int[3];
            for (int v=0; v<3; v++) {
                xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
                yw[v] = vertices[triangles[t*3+v]*3+1];
                zw[v] = vertices[triangles[t*3+v]*3+2];
                x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
                y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
            }

            int bbminx = width-1; // screen bounding box for the triangle to rasterize
            int bbminy = height-1;
            int bbmaxx = 0;
            int bbmaxy = 0;
            for (int v=0; v<3; v++) {
                bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
                bbminy = Math.max(0, Math.min(bbminy, y[v]));
                bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
                bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
            }
            
            int intensity;
            double[] normal = triangle_normal(xw, yw, zw);  // it is already normalized 
            double[] light = {0.3, 0.3, 1. };
            	double normL = Math.sqrt(light[0]*light[0] + light[1]*light[1] + light[2]*light[2]);
            light[0] = light[0]/normL;
            light[1] = light[1]/normL;
            light[2] = light[2]/normL;
            //Vec3f r = (n*(n*l*2.f) - l).normalize();   // reflected light
            double nl = dot_product(normal,light)*2.;
            double[] ncoeff = {normal[0]*nl,  normal[1]*nl, normal[2]*nl};
            double[] reflected = {ncoeff[0]-light[0], ncoeff[1]-light[1],ncoeff[2]-light[2] };
            double norm = Math.sqrt(reflected[0]*reflected[0] + reflected[1]*reflected[1] + reflected[2]*reflected[2]);
            reflected[0] = reflected[0]/norm;
            reflected[1] = reflected[1]/norm;
            reflected[2] = reflected[2]/norm;
   
            double diffuse = Math.max(0, dot_product(normal, light));
            
            float spec = (float) Math.pow(Math.max(dot_product(reflected, light), 0.)  , 5) ;
            //float spec = (float) Math.pow(Math.max(reflected[2], 0.)  , 5) ; // supposes the veiwDir = (0,0,1) !
         
            
            //intensity = (int) Math.min(255, 255*diffuse) ;
            intensity = (int)  Math.min(255, (5 + 255*(diffuse + .6*spec))) ;    
            	//std::min<float>(5 + c[i]*(diff + .6*spec), 255);
            int color = new Color(intensity, intensity, intensity).getRGB(); //color for the current TRIANGLE
            
            try { // non-ivertible matrix (can happen if a triangle is degenerate)
                for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                    for (int py=bbminy; py<=bbmaxy; py++) {
                        double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                        if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                        double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                        if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                        zbuffer[px][py] = pz;
   
                        image.setRGB(px, py, color);
                    }
                }
            } catch (IllegalStateException ex) {}
        }//end for triangles
        
	}
	
//*******************************************************************************************************************************************
public void SpecularShadingNormalMapping()
{		
	try {
		loadTextureCoords("african_head.obj");
	} 
	catch (FileNotFoundException e) {e.printStackTrace();} 
	catch (IOException e) {e.printStackTrace();}
	
	BufferedImage texture = null;
	int TexWidth=0; 
	int TexHeight=0; 
	try
    {
      // the line that reads the texture image file
	  texture = ImageIO.read(new File("african_head_diffuse.png"));
      System.out.println("Texture : height = "+texture.getHeight() );
      System.out.println("Texture : width  = "+texture.getWidth() );
      TexWidth = texture.getWidth();
      TexHeight = texture.getHeight();
      // work with the image here ...
      //image = texture;
      
    } 
    catch (IOException e) { e.printStackTrace();}
	
	//african_head_nm.png
	BufferedImage normalMap = null;
	int NormalMapWidth=0; 
	int NormalMapHeight=0; 
	try
    {
      // the line that reads the texture image file
	  normalMap = ImageIO.read(new File("african_head_nm.png"));
      System.out.println("Texture : height = "+normalMap.getHeight() );
      System.out.println("Texture : width  = "+normalMap.getWidth() );
      NormalMapWidth  = normalMap.getWidth();
      NormalMapHeight = normalMap.getHeight();
      // work with the image here ...
      //image = texture;
      
    } 
    catch (IOException e) { e.printStackTrace();}
	
	
	zbuffer = new double[width][height]; // initialize the z-buffer
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            zbuffer[i][j] = -1.;
        }
    }
    
    for (int t=0; t<triangles.length/3; t++) { // iterate through all triangles
        double[] xw = new double[3]; // triangle in world coordinates
        double[] yw = new double[3];
        double[] zw = new double[3];
        int[] x = new int[3]; // triangle in screen coordinates
        int[] y = new int[3]; 
        int[] TV = new int[3]; // triangle texture vertices
        double[] utex = new double[3]; // u tex coords
        double[] vtex = new double[3];
        for (int v=0; v<3; v++) {
            xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
            yw[v] = vertices[triangles[t*3+v]*3+1];
            zw[v] = vertices[triangles[t*3+v]*3+2];
            x[v] = (int)( width*(xw[v]+1.)/2.+.5); // world-to-screen transformation
            y[v] = (int)(height*(1.-yw[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
            
            TV[v] = trianglesUV[t*3+v];
            utex[v] = uv[trianglesUV[t*3+v]*3+0];
            vtex[v] = uv[trianglesUV[t*3+v]*3+1];
        }

        int bbminx = width-1; // screen bounding box for the triangle to rasterize
        int bbminy = height-1;
        int bbmaxx = 0;
        int bbmaxy = 0;
        for (int v=0; v<3; v++) {
            bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
            bbminy = Math.max(0, Math.min(bbminy, y[v]));
            bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
            bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
        }
        
        double[] light = {0.5, 0.5, 1. };
    	double normL = Math.sqrt(light[0]*light[0] + light[1]*light[1] + light[2]*light[2]);
    	light[0] = light[0]/normL;
    	light[1] = light[1]/normL;
    	light[2] = light[2]/normL;
        
        try { // non-ivertible matrix (can happen if a triangle is degenerate)
            for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                for (int py=bbminy; py<=bbmaxy; py++) {
                    double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                    if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                    double pz = coord[0]*zw[0] + coord[1]*zw[1] + coord[2]*zw[2]; // compute the depth of the fragment
                    if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                    zbuffer[px][py] = pz;
                    
                    
                  //UV coords interpolation  
                    double currentU = (coord[0]*utex[0] + coord[1]*utex[1] + coord[2]*utex[2]);
                    double currentV = (coord[0]*vtex[0] + coord[1]*vtex[1] + coord[2]*vtex[2]);                        
                    int A = (int)(currentU*TexWidth);
    	            int B = (int)((1-currentV)*TexHeight); // v flipped !
    	            Color col = new Color( texture.getRGB( A, B ) ); // color of the current pixel (from the texture) 
    	            
    	            double[] NPix = new double[3]; // normal ot the current pixel
    	            Color colNormalMap = new Color( normalMap.getRGB( A, B ) ); // color from the normal map texture 
    	            //color in [0,255] -> we need [-1,1]
    	            NPix[0] = colNormalMap.getRed()-128;
    	            NPix[1] = colNormalMap.getGreen()-128;
    	            NPix[2] = colNormalMap.getBlue()-128;       	            
    	            double norm = Math.sqrt(NPix[0]*NPix[0] + NPix[1]*NPix[1] +  NPix[2]*NPix[2]); 
                    NPix[0] = NPix[0]/norm;
                    NPix[1] = NPix[1]/norm;
                    NPix[2] = NPix[2]/norm;
                    
                    double nl = dot_product(NPix,light)*2.;
                    double[] ncoeff = {NPix[0]*nl,  NPix[1]*nl, NPix[2]*nl};
                    double[] reflected = {ncoeff[0]-light[0], ncoeff[1]-light[1],ncoeff[2]-light[2] };
                    double norm2 = Math.sqrt(reflected[0]*reflected[0] + reflected[1]*reflected[1] + reflected[2]*reflected[2]);
                    reflected[0] = reflected[0]/norm2;
                    reflected[1] = reflected[1]/norm2;
                    reflected[2] = reflected[2]/norm2;
           
                    double diffuse = Math.max(0, dot_product(NPix, light));
                    
                    float spec = (float) Math.pow(Math.max(dot_product(reflected, light), 0.)  , 5) ;
                    //float spec = (float) Math.pow(Math.max(reflected[2], 0.)  , 5) ; // supposes the veiwDir = (0,0,1) !
                          
                     
                    int intensityRED =   (int)  Math.min(255, (0 + col.getRed()  *(diffuse + 0.6*spec))) ; 
                    int intensityGREEN = (int)  Math.min(255, (0 + col.getGreen()*(diffuse + 0.6*spec))) ; 
                    int intensityBLUE =  (int)  Math.min(255, (0 + col.getBlue() *(diffuse + 0.6*spec))) ; 
                    
//                    int intensityRED =   (int)  Math.min(255, (5 + 255*(diffuse + .6*spec))) ; 
//                    int intensityGREEN = (int)  Math.min(255, (5 + 255*(diffuse + .6*spec))) ; 
//                    int intensityBLUE =  (int)  Math.min(255, (5 + 255*(diffuse + .6*spec))) ;    	                
                    
                    int color = new Color(intensityRED, intensityGREEN, intensityBLUE).getRGB(); 
                    image.setRGB(px, py, color);
                    
                }
            }
        } catch (IllegalStateException ex) {}
    }//end for triangles
    
}


//******************************************************************************************************************************************
void FlatWithShadows_Pass1()
{
	
    zbufferShadow = new double[width][height]; // initialize the z-buffer
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            zbufferShadow[i][j] = -1.;
        }
    }
    double w=1;
    
//    double[][] Trans = {
//			{Math.sqrt(3.)/Math.sqrt(6.) , 0,             -Math.sqrt(3.)/Math.sqrt(6.), 0},
//			{-3/Math.sqrt(54.),             6/Math.sqrt(54.),              -3/Math.sqrt(54.),             0},
//			{Math.sqrt(3.)/3,  Math.sqrt(3.)/3 , Math.sqrt(3.)/3, 0},
//			{0,0,0,1}
//	};
    
    //LookAt with camera at (1,1,1)
    double[][] LookAt = {
			{Math.sqrt(3.)/Math.sqrt(6.),   0,                 -Math.sqrt(3.)/Math.sqrt(6.), 0},
			{-3/Math.sqrt(54.),             6/Math.sqrt(54.),  -3/Math.sqrt(54.),            0},
			{Math.sqrt(3.)/3,               Math.sqrt(3.)/3 ,   Math.sqrt(3.)/3,             0},
			{0,                             0,                  0,                           1}
	}; 
    
    double[][] ViewPort = {
    		{width/2.,    0.,        0.,    width/2.},
    		{0.      ,  -height/2.,  0,     height/2.},
    		{0.,          0.,        1.,    1.},
    		{0.,          0.,        0.,    1.}
    		
    }; 
    // ViewPort*Projection*LookAt*Model transforms the object's coordinates into the (framebuffer) screen space
    // Projection = ortho; LookAt with camera at (1,1,1); Model is Id4 (unique object into the scene)
    
    M =  matrix_product(ViewPort, LookAt); 
    //double[][] M = ViewPort;
    
    for (int t=0; t< triangles.length/3; t++) { // iterate through all triangles
        double[] xw = new double[3]; // triangle in world coordinates
        double[] yw = new double[3];
        double[] zw = new double[3];
        int[] x = new int[3]; // triangle in screen coordinates
        int[] y = new int[3];
        
        double[] xS = new double[3]; // triangle in ViewPort*LookAt = M coordinates
        double[] yS = new double[3];
        double[] zS = new double[3];
        
      
        for (int v=0; v<3; v++) {
            xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
            yw[v] = vertices[triangles[t*3+v]*3+1];
            zw[v] = vertices[triangles[t*3+v]*3+2];
                        
//            xS[v] = Trans[0][0]*xw[v] + Trans[0][1]*yw[v] + Trans[0][2]*zw[v] + Trans[0][3];
//            yS[v] = Trans[1][0]*xw[v] + Trans[1][1]*yw[v] + Trans[1][2]*zw[v] + Trans[1][3];
//            zS[v] = Trans[2][0]*xw[v] + Trans[2][1]*yw[v] + Trans[2][2]*zw[v] + Trans[2][3];
//            w     = Trans[3][0]*xw[v] + Trans[3][1]*yw[v] + Trans[3][2]*zw[v] + Trans[3][3];
//            xS[v] = xS[v]/w;
//            yS[v] = yS[v]/w;          
//            x[v] = (int)( width*(xS[v]+1.)/2.+.5); // world-to-screen transformation
//            y[v] = (int)(height*(1.-yS[v])/2.+.5); // y is flipped to get a "natural" y orientation (origin in the bottom left corner)
//            
            
            xS[v] = M[0][0]*xw[v] + M[0][1]*yw[v] + M[0][2]*zw[v] + M[0][3];
            yS[v] = M[1][0]*xw[v] + M[1][1]*yw[v] + M[1][2]*zw[v] + M[1][3];
            zS[v] = M[2][0]*xw[v] + M[2][1]*yw[v] + M[2][2]*zw[v] + M[2][3];
            w     = M[3][0]*xw[v] + M[3][1]*yw[v] + M[3][2]*zw[v] + M[3][3];
            xS[v] = xS[v]/w;
            yS[v] = yS[v]/w; 
            zS[v] = zS[v]/w; 
            x[v] = (int)xS[v];
            y[v] = (int)yS[v];                
        }
        

        int bbminx = width-1; // screen bounding box for the triangle to rasterize
        int bbminy = height-1;
        int bbmaxx = 0;
        int bbmaxy = 0;
        for (int v=0; v<3; v++) {
            bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
            bbminy = Math.max(0, Math.min(bbminy, y[v]));
            bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
            bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
        }
        
        
        double[] light = {1., 1., 1. };
        //double[] light = {1., 1., 0. };
    	double normL = Math.sqrt(light[0]*light[0] + light[1]*light[1] + light[2]*light[2]);
    	light[0] = light[0]/normL;
    	light[1] = light[1]/normL;
    	light[2] = light[2]/normL;
    	 
        try { // non-ivertible matrix (can happen if a triangle is degenerate)
            for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                for (int py=bbminy; py<=bbmaxy; py++) {
                    double[] coord = barycentric_coords((int)xS[0], (int)yS[0],  (int)xS[1], (int)yS[1], (int)xS[2], (int)yS[2], px, py);
                    if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                    double pz = coord[0]*zS[0] + coord[1]*zS[1] + coord[2]*zS[2]; // compute the depth of the fragment
                    if (zbufferShadow[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                    zbufferShadow[px][py] = pz;
          
                    //px, py and pz : coords in the light space    
                    //System.out.println("px = " + px + " py = "+py);
                    //System.out.println("pz       = " + pz);
                    
                  /* double[] normal = triangle_normal(xw, yw, zw); 
                    double diffuse = Math.max(0, dot_product(normal, light));
                    int intensity = (int) Math.min(255, 255*diffuse) ; 
                    int color = new Color(intensity, intensity, intensity).getRGB(); 
                    image.setRGB(px, py, color); */
                }
            }
        } catch (IllegalStateException ex) {}
    }//end for triangles
    
    
    this.saveZShadowBuffer();
    
}

//---------------------------------------------------------------------------------------------------------------------------------------
void FlatWithShadows_Pass2()
{
	
    zbuffer = new double[width][height]; // initialize the z-buffer
    for (int i=0; i<width; i++) {
        for (int j=0; j<height; j++) {
            zbuffer[i][j] = -1.;
        }
    } 
    
    double w=1;
    
    double[][] ViewPort = {
    		{width/2.,    0.     ,   0.,    width/2.},
    		{0.      ,  -height/2.,  0,     height/2.},
    		{0.,          0.,        1.,    1.},
    		{0.,          0.,        0.,    1.}
    		
    }; 
    // ViewPort*Projection*LookAt*Model transforms the object's coordinates into the (framebuffer) screen space
    // Projection = ortho; LookAt with camera at (0,0,1) gives Id4; Model is Id4 (unique object into the scene)
    
  //matrix_inverse(ViewPort); 
    double[][] ViewPortInverse = { {0.0025, 0,      0.,   -1.},
    		                       {0.,    -0.0025, 0.,    1.},
    		                       {0.,     0.,     1,    -1.},
    		                       {0.,     0.,     0.,    1.}};
    
    double[][] Magic = matrix_product(M, ViewPortInverse);
    
    double[] light = {1., 1., 1. };
    //double[] light = {1., 1., 0. };
	double normL = Math.sqrt(light[0]*light[0] + light[1]*light[1] + light[2]*light[2]);
	light[0] = light[0]/normL;
	light[1] = light[1]/normL;
	light[2] = light[2]/normL;
    
    for (int t=0; t< triangles.length/3; t++) { // iterate through all triangles
        double[] xw = new double[3]; // triangle in world coordinates
        double[] yw = new double[3];
        double[] zw = new double[3];
        int[] x = new int[3]; // triangle in screen coordinates
        int[] y = new int[3]; 
        
        double[] xS = new double[3]; // triangle in screen (ViewPort) coordinates
        double[] yS = new double[3];
        double[] zS = new double[3];
              
        for (int v=0; v<3; v++) {
            xw[v] = vertices[triangles[t*3+v]*3+0]; // world coordinates
            yw[v] = vertices[triangles[t*3+v]*3+1];
            zw[v] = vertices[triangles[t*3+v]*3+2];
            
            xS[v] = ViewPort[0][0]*xw[v] + ViewPort[0][1]*yw[v] + ViewPort[0][2]*zw[v] + ViewPort[0][3];
            yS[v] = ViewPort[1][0]*xw[v] + ViewPort[1][1]*yw[v] + ViewPort[1][2]*zw[v] + ViewPort[1][3];
            zS[v] = ViewPort[2][0]*xw[v] + ViewPort[2][1]*yw[v] + ViewPort[2][2]*zw[v] + ViewPort[2][3];
            w     = ViewPort[3][0]*xw[v] + ViewPort[3][1]*yw[v] + ViewPort[3][2]*zw[v] + ViewPort[3][3];
            xS[v] = xS[v]/w;
            yS[v] = yS[v]/w;  
            zS[v] = zS[v]/w; 
            
            x[v] = (int)xS[v]; //screen coords
            y[v] = (int)yS[v];            
            
        }
        
        int bbminx = width-1; // screen bounding box for the triangle to rasterize
        int bbminy = height-1;
        int bbmaxx = 0;
        int bbmaxy = 0;
        for (int v=0; v<3; v++) {
            bbminx = Math.max(0, Math.min(bbminx, x[v])); // note that the bounding box is clamped to the actual screen size
            bbminy = Math.max(0, Math.min(bbminy, y[v]));
            bbmaxx = Math.min(width-1,  Math.max(bbmaxx, x[v]));
            bbmaxy = Math.min(height-1, Math.max(bbmaxy, y[v]));
        }
        
            	       
        try { // non-ivertible matrix (can happen if a triangle is degenerate)
            for (int px=bbminx; px<=bbmaxx; px++) { // rasterize the bounding box
                for (int py=bbminy; py<=bbmaxy; py++) {
                    //double[] coord = barycentric_coords(x[0], y[0], x[1], y[1], x[2], y[2], px, py);
                	double[] coord = barycentric_coords((int)xS[0], (int)yS[0],  (int)xS[1], (int)yS[1], (int)xS[2], (int)yS[2], px, py);
                    if (coord[0]<-0.01 || coord[1]<-0.01 || coord[2]<-0.01) continue; // discard the point outside the triangle
                    double pz = coord[0]*zS[0] + coord[1]*zS[1] + coord[2]*zS[2]; // compute the depth of the fragment
                    if (zbuffer[px][py]>pz) continue; // discard the fragment if it lies behind the z-buffer
                    zbuffer[px][py] = pz;
                    
                  //px, py and pz : coords in the screen space
                    double[] screenCoords = {px, py, pz, 1};
                    double[] shadowScreenCoords = {0,0,0,1};
                    
                    shadowScreenCoords[0] = Magic[0][0]*screenCoords[0] + Magic[0][1]*screenCoords[1] + Magic[0][2]*screenCoords[2] + Magic[0][3]*screenCoords[3];
                    shadowScreenCoords[1] = Magic[1][0]*screenCoords[0] + Magic[1][1]*screenCoords[1] + Magic[1][2]*screenCoords[2] + Magic[1][3]*screenCoords[3];
                    shadowScreenCoords[2] = Magic[2][0]*screenCoords[0] + Magic[2][1]*screenCoords[1] + Magic[2][2]*screenCoords[2] + Magic[2][3]*screenCoords[3];
                    w                     = Magic[3][0]*screenCoords[0] + Magic[3][1]*screenCoords[1] + Magic[3][2]*screenCoords[2] + Magic[3][3]*screenCoords[3];
                    shadowScreenCoords[0] = shadowScreenCoords[0]/w;
                    shadowScreenCoords[1] = shadowScreenCoords[1]/w; 
                    shadowScreenCoords[2] = shadowScreenCoords[2]/w; 
                    
                   
                    int a = (int)shadowScreenCoords[0];
                    int b = (int)shadowScreenCoords[1];

                    	double shadow = zbufferShadow[a][b];
                    	//System.out.println(" a = "+ a + " b = "+  b + " sh :" + shadow);
                    	//System.out.println("px = " + px + " py = "+py);
                        //System.out.println("pz       = " + pz);
                        //System.out.println("shadow z = " + shadowScreenCoords[2]);
                    
                    
                    	double[] normal = triangle_normal(xw, yw, zw); 
                    	double diffuse = Math.max(0, dot_product(normal, light));
                    	int intensity = 128;
                    	int color = new Color(128, 128, 128).getRGB();;
                    	if(shadow-.1 > shadowScreenCoords[2]) // the pixel is in the shadow 
                    	{
                    		intensity = (int) Math.min(255, 50*diffuse*(shadow-.1+1)) ; 
                    		//color = new Color(0, 0, 255).getRGB();
                    	}
                    	else // the pixel is lit  
                    	{
                    		intensity = (int) Math.min(255, 50*diffuse*(shadow-.1+2)) ;  
                    		//color = new Color(255, 0, 0).getRGB();
                    	}
                    	
                    	/*if(shadow<0) {
                    		color = new Color(0, 255, 0).getRGB();}*/
                    	 
                    	
                    	color = new Color(intensity, intensity, intensity).getRGB(); 
                    	image.setRGB(px, py, color);
                    
                }
            }
        } catch (IllegalStateException ex) {}
    }//end for triangles
    
    //saveZBuffer();
    
}

//*********************************************************************************************************************************************
/**
 * paint for Graphics
 */
	public void paint(Graphics g) {
		g.drawImage(image, 0, 0, this);
	}

/**
 * Main function 
 * @param args
 */
	public static void main(String[] args)  {
		
		TinyRenderer myRenderer = new TinyRenderer();
		
		try {
			myRenderer.loadObjVerticesTriangles("african_head.obj");
		} 
		catch (FileNotFoundException e) {e.printStackTrace();} 
		catch (IOException e) {e.printStackTrace();}
				
		//myRenderer.drawWireMesh();
		//myRenderer.drawTriangleMesh();
		//myRenderer.drawTriangleMeshZBuffer();
		//myRenderer.saveZBuffer();
		
		//myRenderer.UVInterpolateOnly();
		
		//myRenderer.FlatShading(); // with matrices
		//myRenderer.FlatShading_First(); // first attempt without matrices
		//myRenderer.FlatShadingUVTextureInterpolation();
		
		//myRenderer.GouraudShading();
				//myRenderer.GouraudShadingTexture(); //TO REMOVE	
		//myRenderer.GouraudShadingUVTextureInterpolation();		
		
		//myRenderer.PhongShading();
			//myRenderer.PhongShadingTexture(); // TO REMOVE
		 //myRenderer.PhongShadingUVTextureInterpolation();
		
		 //myRenderer.NormalMapping();
		
		 //myRenderer.SpecularShadingFlat();
		 //myRenderer.SpecularShadingNormalMapping();
		
		myRenderer.FlatWithShadows_Pass1();
		myRenderer.FlatWithShadows_Pass2();

		try {
			ImageIO.write(image, "png", new File("shadow.png"));
		} catch (IOException ex) {ex.printStackTrace();}


		JFrame frame = new JFrame();
		frame.getContentPane().add(myRenderer);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(width, height);
		frame.setVisible(true);
	}
}
