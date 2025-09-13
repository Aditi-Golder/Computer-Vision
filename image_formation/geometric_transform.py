import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread('images/original_image.jpg')
img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
transformed_image_org = cv2.imread('images/transformed_image.jpg')
transformed_image = cv2.cvtColor(transformed_image_org, cv2.COLOR_BGR2RGB)

rows,cols,ch = img.shape
 
print(f"Image dimensions: {rows} x {cols} x {ch}")

# Define source and destination points for affine transformation
pts1 = np.float32([[160,160],[160,1000],[1000,180]])
pts2 = np.float32([[450,250],[750,1100],[850,500]])

print("\n=== TRANSFORMATION ANALYSIS ===")
print(f"Source points (original):")
for i, pt in enumerate(pts1):
    print(f"  Point {i+1}: ({pt[0]:.0f}, {pt[1]:.0f})")

print(f"\nDestination points (transformed):")
for i, pt in enumerate(pts2):
    print(f"  Point {i+1}: ({pt[0]:.0f}, {pt[1]:.0f})")

# Calculate the affine transformation matrix
M = cv2.getAffineTransform(pts1,pts2)

print(f"\nAffine Transformation Matrix:")
print(f"[{M[0,0]:.4f}  {M[0,1]:.4f}  {M[0,2]:.2f}]")
print(f"[{M[1,0]:.4f}  {M[1,1]:.4f}  {M[1,2]:.2f}]")

# Analyze transformation components
# Translation components
tx = M[0, 2]
ty = M[1, 2]

# Scale and rotation components (from the 2x2 upper-left matrix)
a, b = M[0, 0], M[0, 1]
c, d = M[1, 0], M[1, 1]

# Calculate scale factors
scale_x = np.sqrt(a**2 + c**2)
scale_y = np.sqrt(b**2 + d**2)

# Calculate rotation angle (in degrees)
rotation_angle = np.arctan2(c, a) * 180 / np.pi

print(f"\n=== TRANSFORMATION COMPONENTS ===")
print(f"Translation: ({tx:.2f}, {ty:.2f}) pixels")
print(f"Scale factors: X={scale_x:.3f}, Y={scale_y:.3f}")
print(f"Rotation angle: {rotation_angle:.2f} degrees")


# Apply the transformation
dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(131),plt.imshow(img),plt.title('Input')
plt.subplot(132),plt.imshow(transformed_image),plt.title('Transformed Image')
plt.subplot(133),plt.imshow(dst),plt.title('My Output')
plt.savefig("Output/geometric_transform.png", dpi=300)
plt.show()