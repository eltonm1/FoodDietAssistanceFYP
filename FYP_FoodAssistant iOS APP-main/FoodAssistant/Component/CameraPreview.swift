//
//  CameraPreview.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 8/11/2022.
//

import SwiftUI
import AVFoundation
import UIKit

struct CameraPreview: UIViewRepresentable {
    
    let session: AVCaptureSession
    
    func makeUIView(context: Context) -> VideoPreviewView {
        let view: VideoPreviewView = context.coordinator.view
        view.videoPreviewLayer.cornerRadius = 0
        view.videoPreviewLayer.session = session
        switch UIDevice.current.orientation {
        case .landscapeLeft:
            view.videoPreviewLayer.connection?.videoOrientation = .landscapeRight
        case .landscapeRight:
            view.videoPreviewLayer.connection?.videoOrientation = .landscapeLeft
        default:
            view.videoPreviewLayer.connection?.videoOrientation = .portrait
        }
        view.videoPreviewLayer.videoGravity = .resizeAspectFill

        return view
    }

    func updateUIView(_ uiView: VideoPreviewView, context: Context) { }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject {
        let view = VideoPreviewView()
        
        override init() {
            super.init()
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(viewDidOriented),
                name: UIDevice.orientationDidChangeNotification,
                object: nil
            )
        }
        
        deinit {
            NotificationCenter.default.removeObserver(
                self,
                name: UIDevice.orientationDidChangeNotification,
                object: nil
            )
        }
        
        @objc func viewDidOriented() {
            let orientation: UIDeviceOrientation = UIDevice.current.orientation
            if orientation == .portrait {
                view.videoPreviewLayer.connection?.videoOrientation = .portrait
            } else if orientation == .landscapeLeft {
                view.videoPreviewLayer.connection?.videoOrientation = .landscapeRight
            } else if orientation == .landscapeRight {
                view.videoPreviewLayer.connection?.videoOrientation = .landscapeLeft
            } else if orientation == .portraitUpsideDown {
                view.videoPreviewLayer.connection?.videoOrientation = .portraitUpsideDown
            }
        }
    }
    
//    func makeUIView(context: Context) -> UIView {
//        let view = UIView()
//        let layer = AVCaptureVideoPreviewLayer(session: session)
//        let bounds: CGRect = UIScreen.main.bounds
//        layer.frame = CGRect(x: 0, y: 0, width: bounds.width, height: bounds.height)
//        layer.videoGravity = .resizeAspectFill
//        layer.connection?.videoOrientation = .portrait
//        view.layer.addSublayer(layer)
//        return view
//    }
//
//    func updateUIView(_ uiView: UIView, context: Context) {
//
//    }
}
