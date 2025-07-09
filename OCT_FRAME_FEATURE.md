# Enhanced OCT Frame Extraction and Navigation Feature

## Overview

This feature enhances the E2E file processing to extract **ALL OCT frames** from the volume (not just the middle frame) and provides seamless navigation through individual OCT frames with proper caching.

## Key Features

### 1. **Complete OCT Frame Extraction**
- Extracts all frames from OCT volumes (not just middle frame)
- Processes both original and flattened OCT frames
- Maintains backward compatibility with existing functionality

### 2. **Enhanced File Tree Structure**
- **Original OCT Frames**: All individual OCT frames from the volume
- **Flattened OCT Frames**: All flattened OCT frames with enhanced processing
- **Legacy OCT**: Backward compatibility for existing OCT files
- **Fundus/DICOM**: Fundus images (unchanged)

### 3. **Seamless Frame Navigation**
- Click any OCT frame to load it in the viewer
- Automatic frame slider context switching
- Video-like navigation through OCT frame sequences
- Real-time frame information display

### 4. **Advanced Caching System**
- Individual frame caching with CRC-based system
- Memory and disk caching for optimal performance
- Automatic cache cleanup and management

## Backend Implementation

### Enhanced E2E Processing (`process_e2e_file`)

```python
# New data structure
stored_images[key]["left_eye_data"] = {
    "dicom": [],           # Fundus images
    "oct": [],            # Legacy OCT (middle frame only)
    "original_oct": [],   # NEW: All OCT frames
    "flattened_oct": []   # NEW: All flattened OCT frames
}
```

### New API Endpoints

#### 1. Enhanced Tree Data
```
GET /api/get_e2e_tree_data?dicom_file_path=<crc>
```
Returns enhanced tree structure with `original_oct` and `flattened_oct` branches.

#### 2. OCT Frame Metadata
```
GET /api/get_e2e_oct_frames?dicom_file_path=<crc>&eye=<left|right>
```
Returns metadata for all OCT frames including frame numbers, IDs, and display names.

#### 3. Individual OCT Frame Viewing
```
GET /api/view_e2e_oct_frame?dicom_file_path=<crc>&eye=<left|right>&frame_number=<n>&frame_type=<original|flattened>
```
Serves individual OCT frame images with proper caching headers.

## Frontend Implementation

### Enhanced File Tree

The file tree now shows:
- **Original OCT Frames (N)**: Individual frames like `frame_0001.jpg`, `frame_0002.jpg`, etc.
- **Flattened OCT Frames (N)**: Flattened versions of each frame
- **OCT (Legacy) (N)**: Backward compatibility
- **Fundus/DICOM (N)**: Fundus images

### Frame Navigation

#### Click to Load
```javascript
// Click any OCT frame to load it
selectOCTFrame(eye, frameType, frameNumber, frameKey, event)
```

#### Slider Navigation
```javascript
// Automatic slider context switching
switchToOCTFrameMode(viewportNumber, eye, frameType, currentFrame)
```

#### Frame Loading
```javascript
// Load individual OCT frames
loadOCTFrame(viewportNumber, eye, frameNumber, dicomFilePath, frameType)
```

### Enhanced Frame Slider

- Automatically switches to OCT frame mode when OCT frames are selected
- Shows frame-specific information (OCT original/flattened - left/right eye)
- Maintains current frame position when switching between image types

## User Experience Flow

### 1. **Load E2E File**
- User clicks on .e2e file in S3 tree
- Backend processes and extracts ALL OCT frames
- File tree populates with new branches

### 2. **Navigate OCT Frames**
- User clicks on "Original OCT Frames" folder
- Individual frames are displayed (frame_0001.jpg, frame_0002.jpg, etc.)
- User clicks any frame to load it

### 3. **Frame Slider Navigation**
- Frame slider automatically switches to OCT frame mode
- User can slide between all OCT frames like a video
- Frame info shows: "OCT original - left eye - Frame 5 of 128"

### 4. **Switch Between Types**
- User can switch between original and flattened OCT frames
- Frame slider context updates automatically
- Seamless navigation experience

## Caching Implementation

### Memory Cache
- Individual frames stored in `stored_images` with unique keys
- Frame metadata stored for quick access
- Automatic cleanup of old entries

### Disk Cache
- CRC-based caching system
- Hierarchical directory structure (`cache/e2e/<crc>/`)
- Frame images stored as individual files
- Metadata stored in `metadata.pkl`

### Cache Structure
```
cache/
├── e2e/
│   └── <crc>/
│       ├── metadata.pkl
│       ├── frame_0.jpg
│       ├── frame_1.jpg
│       └── ...
```

## Performance Optimizations

### 1. **Parallel Processing**
- OCT frame processing can be parallelized (future enhancement)
- Individual frame caching reduces reprocessing

### 2. **Lazy Loading**
- Frames are loaded on-demand
- Only requested frames are processed and cached

### 3. **Memory Management**
- Automatic cleanup of old cache entries
- Efficient storage of frame metadata

## Testing

### Test Script
Run the test script to verify functionality:
```bash
python test_oct_frames.py
```

### Manual Testing
1. Upload an E2E file through the web interface
2. Check for new "Original OCT Frames" and "Flattened OCT Frames" branches
3. Click individual frames to load them
4. Use frame slider to navigate between frames
5. Switch between original and flattened views

## API Examples

### Get Enhanced Tree Data
```bash
curl "http://localhost:8000/api/get_e2e_tree_data?dicom_file_path=<crc>"
```

### Get OCT Frame Metadata
```bash
curl "http://localhost:8000/api/get_e2e_oct_frames?dicom_file_path=<crc>&eye=left"
```

### View OCT Frame
```bash
curl "http://localhost:8000/api/view_e2e_oct_frame?dicom_file_path=<crc>&eye=left&frame_number=5&frame_type=original"
```

## Future Enhancements

### 1. **Parallel Processing**
- Process OCT frames in parallel for faster extraction
- Background processing with progress updates

### 2. **Advanced Visualization**
- 3D OCT volume rendering
- En-face OCT views
- OCT angiography processing

### 3. **Analysis Tools**
- Automated OCT analysis
- Thickness measurements
- Pathology detection

### 4. **Export Features**
- Export OCT frames to different formats
- Batch processing capabilities
- OCT measurement reports

## Troubleshooting

### Common Issues

1. **No OCT frames visible**
   - Check if E2E file contains OCT data
   - Verify file processing completed successfully
   - Check browser console for errors

2. **Frame slider not working**
   - Ensure OCT frame mode is active
   - Check if frame count is correct
   - Verify frame data is loaded

3. **Cache issues**
   - Clear browser cache
   - Check server cache status
   - Restart server if needed

### Debug Information

Enable debug logging to see detailed information:
```python
logger.setLevel(logging.DEBUG)
```

Check cache status:
```bash
curl "http://localhost:8000/api/cache-status"
```

## Conclusion

This enhanced OCT frame extraction feature provides a complete solution for navigating through OCT volumes with individual frame access, seamless navigation, and proper caching. The implementation maintains backward compatibility while adding powerful new capabilities for OCT image analysis and visualization. 