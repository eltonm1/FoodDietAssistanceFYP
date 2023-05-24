//
//  VideoPreviewView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 8/11/2022.
//

import UIKit
import AVFoundation

class VideoPreviewView: UIView {
    override class var layerClass: AnyClass {
         AVCaptureVideoPreviewLayer.self
    }
    
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        return layer as! AVCaptureVideoPreviewLayer
    }
}
