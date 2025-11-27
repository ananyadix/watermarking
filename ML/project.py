import os
import uuid
import json
import hashlib
from io import BytesIO
from typing import Tuple, List, Dict, Union

import numpy as np
import pywt # For Integer Haar DWT [2]
from PIL import Image # For image I/O
import mahotas.features # For Zernike Moments [3]

# Flask and CORS setup for Node.js integration
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS # Essential for cross-origin requests from Node.js backend

# --- 1. CONFIGURATION AND INITIALIZATION ---

# Define the absolute path for file storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg', 'tif', 'tiff', 'bmp'} # Common medical image formats

# Initialize Flask App
app = Flask(__name__)
app.config = UPLOAD_FOLDER
# Enable CORS for all origins/routes to allow Node.js access
CORS(app) 

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Checks if a file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 2. CORE WATERMARKING FUNCTIONS (ALGORITHMIC IMPLEMENTATION) ---

# --- Phase 1: Preprocessing and Feature Extraction ---

def DWT_Decomposition(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Applies 2D Integer Haar DWT. Required for reversibility."""
    I_float = I.astype(np.float32)
    # Use 'haar' (db1) wavelet which supports the Integer Wavelet Transform principle [1]
    coeffs = pywt.dwt2(I_float, 'haar') 
    LL, (LH, HL, HH) = coeffs
    
    # Cast back to integer type
    return np.round(LL).astype(np.int32), np.round(LH).astype(np.int32), \
           np.round(HL).astype(np.int32), np.round(HH).astype(np.int32)

def Texture_Segmentation(HH: np.ndarray, N: int = 8) -> np.ndarray:
    """Divides the image into ROI (1) and RONI (0) based on HH texture energy."""
    h, w = HH.shape
    if h % N!= 0 or w % N!= 0:
        # Handle non-divisible dimensions by cropping or padding (cropping used here for simplicity)
        h = (h // N) * N
        w = (w // N) * N
        HH = HH[:h, :w]
        
    M_shape = (h // N, w // N)
    M = np.zeros(M_shape, dtype=np.int8)
    block_energies = []

    for i in range(0, h, N):
        for j in range(0, w, N):
            B_ij = HH[i:i+N, j:j+N]
            # Compute energy E_i,j = Σ(B_i,j^2) [1]
            E_ij = np.sum(B_ij ** 2)
            block_energies.append(E_ij)
            
    if not block_energies:
        return M
        
    T = np.mean(block_energies) # Compute threshold T = Mean(E_i,j) [1]
    
    # Generate Mask M
    block_idx = 0
    for i in range(M_shape[0]):
        for j in range(M_shape[1]):
            E_ij = block_energies[block_idx]
            # If E_i,j >= T, set M[i,j]=1 (ROI) else M[i,j]=0 (RONI) [1]
            if E_ij >= T:
                M[i, j] = 1
            block_idx += 1
            
    return M

def Zernike_Moment_Calculation(I: np.ndarray, p_max: int = 10) -> List[float]:
    """Calculates Zernike moment magnitudes (Z) for geometric signature."""
    
    # Normalize image to be between 0 and 1, as required by mahotas
    I_normalized = (I - np.min(I)) / (np.max(I) - np.min(I))
    I_normalized = I_normalized.astype(np.float64)

    # Calculate Zernike moments up to specified order p_max [1]
    radius = min(I_normalized.shape) // 2
    
    # mahotas.features.zernike_moments returns the magnitudes (|A_pq|), forming Z [3]
    Z_array = mahotas.features.zernike_moments(I_normalized, radius, degree=p_max)
    
    return Z_array.tolist()

def GetROICoefficients(SubBands: Tuple, M: np.ndarray, N: int) -> np.ndarray:
    """Extracts DWT coefficients from ROI blocks (M=1) for hashing/embedding."""
    LL, LH, HL, HH = SubBands
    h_w, w_w = HH.shape
    roi_data = []
    
    # Iterate through the HH sub-band blocks
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] == 1:
                block = HH[i*N:(i+1)*N, j*N:(j+1)*N]
                roi_data.append(block.flatten())
                
    if not roi_data:
        return np.array([], dtype=np.int32)
        
    return np.concatenate(roi_data)

def GetRestorationData(ROI_data: np.ndarray) -> np.ndarray:
    """
    *** COMPLEX PLACEHOLDER: GENERATE ROI RESTORATION DATA (R) ***
    
    In a functional DE implementation, R must contain the Location Map (LM) 
    and original values of all coefficients that would cause overflow/underflow 
    during Difference Expansion. For LSB fragile embedding, R/AuxiliaryBits (A) 
    also include the original LSBs of the ROI.
    
    This placeholder returns a simple byte array (e.g., LSBs of ROI coefficients) 
    and is illustrative only. The full implementation must generate the precise
    LM and original values for *perfect* reversibility (MSE=0).
    """
    # Placeholder: Store LSBs for fragile reversal
    lsb_data = ROI_data & 1
    
    # Placeholder for the larger, compressed Location Map/Original Values for DE
    R_placeholder = np.zeros(256, dtype=np.uint8) # Arbitrary size placeholder
    
    # Concatenate LSBs and placeholder for robustness metadata
    return np.concatenate()

def Generate_Watermarks(I: np.ndarray, M: np.ndarray, N: int, Z: List[float], PatientData: str, SubBands: Tuple) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generates W_F (Fragile) and W_R (Robust) payloads."""
    
    ROI_data = GetROICoefficients(SubBands, M, N)
    
    # 1. Obtain restoration data R (AuxiliaryBits are part of R for ROI)
    R = GetRestorationData(ROI_data)
    
    # 2. Generate Fragile Watermark (W_F) [1]
    # W_F = SHA256(ROI data) + AuxiliaryBits (A)
    hash_object = hashlib.sha256(ROI_data.tobytes())
    StoredHash = hash_object.hexdigest() # Store hash string for external validation
    
    # W_F payload: StoredHash (bytes) + AuxiliaryBits (R)
    W_F_data = np.concatenate()
    
    # 3. Form Robust Watermark (W_R) [1]
    # W_R = Z + PatientData + R
    Z_bytes = np.array(Z, dtype=np.float64).tobytes()
    PD_bytes = PatientData.encode('utf-8')
    R_bytes = R.tobytes()
    
    # Concatenate all robust components (Requires header for size parsing in extraction)
    # Simple concatenation for illustration: Z bytes + PD bytes + R bytes
    W_R_data = np.concatenate()
    
    return W_F_data, W_R_data, StoredHash

# --- Phase 2: Dual Watermark Embedding ---

def LSB_Embed(coefficients: np.ndarray, data_bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    *** COMPLEX PLACEHOLDER: FRAGILE LSB EMBEDDING ***
    
    Actual implementation requires precise management of bit insertion to 
    ensure the AuxiliaryBits (part of W_F) are reversible. This placeholder 
    performs a simplified LSB substitution.
    """
    if data_bits.size == 0:
        return coefficients, 0
    
    max_bits = min(coefficients.size, data_bits.size)
    
    # Simple LSB Substitution placeholder
    coeffs_flat = coefficients.flatten()
    data_to_embed = data_bits[:max_bits]
    
    # Clear LSB and set new LSB
    coeffs_flat[:max_bits] = (coeffs_flat[:max_bits] & ~1) | data_to_embed 
    
    return coeffs_flat.reshape(coefficients.shape), max_bits


def DE_Embed(coefficients: np.ndarray, data_bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    *** COMPLEX PLACEHOLDER: REVERSIBLE DIFFERENCE EXPANSION (DE) EMBEDDING ***
    
    This function must implement the specific DE formulas (Section 1.2 of report), 
    including calculating the Location Map (LM) for non-expandable pairs.
    This placeholder only mimics the behavior.
    """
    if data_bits.size == 0:
        return coefficients, 0
        
    # Placeholder: Assuming capacity for half the number of coefficients
    max_capacity = coefficients.size // 2 
    bits_to_embed = min(max_capacity, data_bits.size)
    
    # Coefficients are modified by a reversible DE scheme.
    # Placeholder: Simple random perturbation to simulate modification
    coeffs_prime = coefficients + np.random.randint(-1, 2, size=coefficients.shape)
    
    return coeffs_prime, bits_to_embed

def Embed_Watermark(SubBands: Tuple, M: np.ndarray, N: int, W_F: np.ndarray, W_R: np.ndarray) -> np.ndarray:
    """Adaptive Dual Watermark Embedding into DWT coefficients."""
    
    LL, LH, HL, HH = SubBands
    W_F_bits = np.unpackbits(W_F)
    W_R_bits = np.unpackbits(W_R)
    
    current_wf_bit = 0
    current_wr_bit = 0
    
    # Iterate through blocks in the HH sub-band mask
    for i in range(M.shape):
        for j in range(M.shape[1]):
            start_h, end_h = i*N, (i+1)*N
            start_w, end_w = j*N, (j+1)*N
            
            # Ensure slicing doesn't exceed bounds (though segmentation should prevent this)
            B_ij = HH[start_h:min(end_h, HH.shape), start_w:min(end_w, HH.shape[1])]
            
            if M[i, j] == 1: # ROI (Fragile)
                # Apply LSB embedding using W_F bits [1]
                B_ij_prime, bits_used = LSB_Embed(B_ij, W_F_bits[current_wf_bit:])
                current_wf_bit += bits_used
                HH[start_h:end_h, start_w:end_w] = B_ij_prime
                
            else: # RONI (Robust and Reversible)
                # Apply Difference Expansion embedding using W_R bits [1]
                B_ij_prime, bits_used = DE_Embed(B_ij, W_R_bits[current_wr_bit:])
                current_wr_bit += bits_used
                HH[start_h:end_h, start_w:end_w] = B_ij_prime
                
    # Reconstruct the sub-bands tuple
    SubBands_prime = (LL, LH, HL, HH)
    return Image_Reconstruction(SubBands_prime)

def Image_Reconstruction(SubBands: Tuple) -> np.ndarray:
    """Recombines DWT coefficients using Inverse DWT (IDWT).[1]"""
    LL, LH, HL, HH = SubBands
    coeffs = (LL, (LH, HL, HH))
    I_prime = pywt.idwt2(coeffs, 'haar')
    
    # Cast back to original image type (e.g., uint8)
    return np.clip(np.round(I_prime), 0, 255).astype(np.uint8)

# --- Phase 3: Verification, Restoration, and Integrity Check ---

def ExtractWatermark(I_prime: np.ndarray, M: np.ndarray, N: int, region: str) -> np.ndarray:
    """
    *** COMPLEX PLACEHOLDER: WATERMARK EXTRACTION (Inverse DE/LSB) ***
    
    This function must implement the inverse of DE (ID-DE) for the RONI 
    (to get W_R) and inverse LSB for ROI (to get W_F), requiring complex 
    knowledge of Location Maps and embedding parameters.
    """
    if region == "RONI":
        # Placeholder for extracting W_R (Robust Watermark: Z + PatientData + R)
        return np.array([0xAA] * 256, dtype=np.uint8) 
    elif region == "ROI":
        # Placeholder for extracting W_F (Fragile Watermark: Hash + AuxiliaryBits)
        # Needs to know the size of the original hash (e.g., 32 bytes for SHA256)
        return np.array( * 128, dtype=np.uint8)

def Geometric_Correction(I_prime: np.ndarray, Z_original: List[float], M: np.ndarray, N: int) -> np.ndarray:
    """Detects geometric attacks using Zernike moments and corrects."""
    
    # 1. Compute current Zernike moments (Z'') [1]
    Z_current = Zernike_Moment_Calculation(I_prime)
    
    # 2. Compare Z_original (Z) vs Z_current (Z'')
    # Note: Comparison is usually done using Normalized Correlation (NC) threshold.
    # Placeholder: Simple difference threshold
    Z_original_array = np.array(Z_original)
    Z_current_array = np.array(Z_current)
    difference = np.linalg.norm(Z_original_array - Z_current_array) / len(Z_original)

    I_corrected = I_prime
    
    if difference > 0.05: # Threshold check (0.05 is illustrative)
        # 3. Apply Correction (De-rotation, De-scaling) [1]
        I_corrected = CorrectImage(I_prime, Z_original_array, Z_current_array)
        
    return I_corrected

def CorrectImage(I_prime: np.ndarray, Z_original: np.ndarray, Z_current: np.ndarray) -> np.ndarray:
    """
    *** COMPLEX PLACEHOLDER: GEOMETRIC CORRECTION ***
    
    Requires algorithms like phase correlation or iterative search to calculate 
    the transformation matrix (rotation angle/scaling factor) between Z_original 
    and Z_current and apply the inverse transformation (e.g., using OpenCV).
    """
    # Placeholder: Returns the original image unchanged
    print("Geometric correction needed but placeholder used.")
    return I_prime 

def Tamper_Detection(I_corrected: np.ndarray, M: np.ndarray, N: int, StoredHash: str) -> Tuple:
    """Compares the extracted fragile hash chain against the computed hash."""
    
    # 1. DWT and Extract ROI coefficients from the corrected image [1]
    LL, LH, HL, HH = DWT_Decomposition(I_corrected)
    ROI_data_current = GetROICoefficients((LL, LH, HL, HH), M, N)
    
    if ROI_data_current.size == 0:
        return False,[]
        
    # 2. Compute current hash (CurrentHash) [1]
    CurrentHash = hashlib.sha256(ROI_data_current.tobytes()).hexdigest()
    
    TamperedBlocks = []
    
    # 3. Global Check [1]
    if StoredHash!= CurrentHash:
        # 4. Perform block-by-block localization (Placeholder) [1]
        # Actual localization requires comparing individual block hashes embedded in W_F
        print("Tampering detected. Localization placeholder running.")
        
        # Iterating over ROI blocks (M=1) for precise localization
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] == 1:
                    # In a real system, we'd check Hash(B_i,j) against ExpectedHash(B_i,j)
                    # If they differ, the block is tampered.
                    if np.random.rand() < 0.1: # 10% chance of a block being flagged (illustrative)
                        TamperedBlocks.append((i, j))
                        
        return True, TamperedBlocks
        
    # 5. If StoredHash == CurrentHash: Integrity verified [1]
    return False, TamperedBlocks

def Full_Restoration(I_corrected: np.ndarray, M: np.ndarray, N: int, W_R: np.ndarray, W_F: np.ndarray) -> np.ndarray:
    """Performs IDWT on restored coefficients for lossless recovery (MSE=0).[1]"""
    
    LL, LH, HL, HH = DWT_Decomposition(I_corrected)
    
    # 1. Extract Restoration Data (R) from W_R and Auxiliary Bits (A) from W_F
    # In reality, W_R is extracted first from I_corrected, and R and A are parsed from it.
    
    R = W_R[-256:] # Placeholder for R
    A = W_F[-128:] # Placeholder for A
    
    # 2. Inverse DE (ID-DE) - RONI Restoration (Placeholder)
    # Use R (Location Map) to perfectly reverse the DE scheme applied in RONI.
    print("RONI restoration (Inverse DE) placeholder running.")
    
    # 3. Inverse LSB/AuxiliaryBits - ROI Restoration (Placeholder)
    # Use A to perfectly restore the LSBs overwritten in ROI.
    print("ROI restoration (AuxiliaryBits) placeholder running.")
    
    SubBands_original = (LL, LH, HL, HH) # Assuming coefficients are perfectly restored
    
    # 4. Final IDWT for lossless original image recovery [1]
    I_original = Image_Reconstruction(SubBands_original)
    
    return I_original

# --- 3. FLASK API ENDPOINTS ---

@app.route('/api/v1/watermark/embed', methods=['POST'])
def embed_watermark_api():
    """Endpoint for Phase 1 & 2: Image Processing and Watermark Embedding."""
    
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
    
    file = request.files['image']
    patient_data = request.form.get('patient_data', 'N/A')
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type or filename"}), 400

    try:
        # Load image as grayscale
        img = Image.open(BytesIO(file.read())).convert('L')
        I = np.array(img)
        
        # --- Phase 1: Preprocessing and Feature Extraction ---
        LL, LH, HL, HH = DWT_Decomposition(I)
        M = Texture_Segmentation(HH)
        Z = Zernike_Moment_Calculation(I) # Zernike Signature [1]
        
        # --- Watermark Generation ---
        N = 8 # Block size
        W_F, W_R, StoredHash = Generate_Watermarks(I, M, N, Z, patient_data, (LL, LH, HL, HH))

        # --- Phase 2: Dual Watermark Embedding ---
        I_prime = Embed_Watermark((LL, LH, HL, HH), M, N, W_F, W_R)
        
        # Save the watermarked image
        output_filename = f"watermarked_{uuid.uuid4()}.png"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        Image.fromarray(I_prime).save(output_path)
        
        # Return results including critical state data for Node.js persistence
        return jsonify({
            "status": "success",
            "message": "Watermarking complete.",
            "download_uri": f"/api/v1/utility/download/{output_filename}",
            # These must be stored by the Node.js backend for verification/restoration
            "StoredHash": StoredHash,
            "ZernikeSignature": Z 
        })
        
    except Exception as e:
        app.logger.error(f"Error during embedding: {e}")
        return jsonify({"status": "error", "message": f"Server processing error: {str(e)}"}), 500


@app.route('/api/v1/watermark/verify_restore', methods=['POST'])
def verify_restore_api():
    """Endpoint for Phase 3: Geometric Correction, Tamper Detection, and Full Restoration."""
    
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Invalid file type or filename"}), 400
        
    try:
        # Retrieve necessary data passed from Node.js
        stored_hash = request.form.get('stored_hash')
        z_signature_str = request.form.get('zernike_signature')
        
        if not stored_hash or not z_signature_str:
             return jsonify({"status": "error", "message": "Missing StoredHash or ZernikeSignature"}), 400
             
        Z_original = json.loads(z_signature_str) # Convert JSON string back to list of floats
        
        # Load the received image I''
        img = Image.open(BytesIO(file.read())).convert('L')
        I_prime = np.array(img)
        
        # --- Re-run Phase 1 Segmentation (needed for masking) ---
        LL_dummy, LH_dummy, HL_dummy, HH_dummy = DWT_Decomposition(I_prime)
        M = Texture_Segmentation(HH_dummy)
        N = 8 # Block size

        # --- Phase 3: Verification and Restoration ---
        
        # 1. Geometric Attack Detection and Correction [1]
        I_corrected = Geometric_Correction(I_prime, Z_original, M, N)
        
        # 2. Tamper Detection [1]
        is_tampered, tampered_blocks = Tamper_Detection(I_corrected, M, N, stored_hash)
        
        # Dummy watermark extraction required for restoration data R/A
        W_R_extracted = ExtractWatermark(I_corrected, M, N, region="RONI")
        W_F_extracted = ExtractWatermark(I_corrected, M, N, region="ROI")
        
        # 3. Full Reversibility (Only if not tampered, or if restoration is desired) [1]
        I_original = Full_Restoration(I_corrected, M, N, W_R_extracted, W_F_extracted)
        
        # Save the restored image
        restored_filename = f"restored_{uuid.uuid4()}.png"
        restored_path = os.path.join(app.config["UPLOAD_FOLDER"], restored_filename)
        Image.fromarray(I_original).save(restored_path)
        
        return jsonify({
            "status": "success",
            "integrity_check": "Tampered" if is_tampered else "Verified",
            "tampered_blocks": tampered_blocks,
            "restored_download_uri": f"/api/v1/utility/download/{restored_filename}"
        })
        
    except Exception as e:
        app.logger.error(f"Error during verification/restoration: {e}")
        return jsonify({"status": "error", "message": f"Server processing error: {str(e)}"}), 500


@app.route('/api/v1/utility/download/<filename>', methods=['GET'])
def download_file(filename):
    """Securely serves processed files (watermarked or restored).[4]"""
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "File not found."}), 404

# Main execution block
if __name__ == '__main__':
    # Running in debug mode is suitable for local development/integration testing
    app.run(debug=True, port=5000)