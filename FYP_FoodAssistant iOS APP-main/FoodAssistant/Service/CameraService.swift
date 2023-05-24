//
//  CameraService.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 8/11/2022.
//

import Foundation
import UIKit
import AVFoundation
import CoreML
import Vision
import SwiftUI

public class CameraService: NSObject, Identifiable, ObservableObject {
    @Published var flashMode: AVCaptureDevice.FlashMode = .off
    @Published var alertError: AlertError? = nil
    @Published var isCameraUnavailable = true
    @Published var photo: Photo?
    @Published var willCapturePhoto = false
    @Published var shouldShowSpinner = false
    
    let sampleBufferQueue = DispatchQueue.global(qos: .userInteractive)
    let session = AVCaptureSession()
    let sessionQueue = DispatchQueue(label: UUID().uuidString)
    let photoOutput = AVCapturePhotoOutput()
    
    var isSessionRunning = false
    var isConfigured = false
    var setupResult: SessionSetupResult = .success
    var inProgressPhotoCaptureDelegates = [Int64: PhotoCaptureProcessor]()
    
    @objc dynamic var videoDeviceInput: AVCaptureDeviceInput!
    
    // MARK: Device Configuration Properties
    let videoDeviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .builtInDualCamera, .builtInTrueDepthCamera], mediaType: .video, position: .unspecified)
    
    var keyValueObservations = [NSKeyValueObservation]()
    
    override public init() {
        super.init()
        
        // Disable the UI. Enable the UI later, if and only if the session starts running.
        DispatchQueue.main.async {
            self.isCameraUnavailable = true
        }
    }
    
    /// Setup the capture session.
    /// 
    /// The session is configured on an internal session queue. 
    /// This ensures that the main queue isn't blocked.
    /// 
    /// - Parameters:
    ///   - onComplete: An optional closure to be executed when the session is configured
    ///   - additionalInput: An optional closure to be executed during the session configuration
    ///     process. This is useful for adding additional inputs to the session.
    ///     The closure should not throw any errors, should not start the session, and should
    ///     not begin or commit the configuration. If there is any configuration that needs to
    ///     modify the capture session, perform it in the session queue provided by the closure
    ///   - additionalOutput: An optional closure to be executed during the session configuration
    ///     process. This is useful for adding additional changes to the `AVCaptureVideoDataOutput` of the session.
    ///     The closure should not throw any errors, should not start the session, and should
    ///     not begin or commit the configuration. If there is any configuration that needs to
    ///     modify the capture session, perform it in the session queue provided by the closure.
    ///     If there is any configuration that needs to be performed on the output, perform it in the
    ///     `AVCaptureVideoDataOutput` provided in the closure
    ///
    public func configure(
        onComplete: (() -> Void)? = nil, 
        additionalInput: ((AVCaptureSession, DispatchQueue) -> Void)? = nil,
        additionalOutput: ((AVCaptureSession, AVCaptureVideoDataOutput, DispatchQueue) -> Void)? = nil
        ) {
        /// In general, it's not safe to mutate an AVCaptureSession or any of its
        /// inputs, outputs, or connections from multiple threads at the same time.
        ///  
        /// Don't perform these tasks on the main queue because
        /// AVCaptureSession.startRunning() is a blocking call, which can
        /// take a long time. Dispatch session setup to the sessionQueue, so
        /// that the main queue isn't blocked, which keeps the UI responsive.
        sessionQueue.async { [weak self] in
            self?.configureSession(
                onComplete: onComplete,
                additionalInput: additionalInput,
                additionalOutput: additionalOutput
            )
        }
    }
    
    // MARK: Checks for permisions, setup obeservers and starts running session
    public func checkForPermissions() {
        /*
         Check the video authorization status. Video access is required and audio
         access is optional. If the user denies audio access, AVCam won't
         record audio during movie recording.
         */
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            // The user has previously granted access to the camera.
            break
        case .notDetermined:
            /*
             The user has not yet been presented with the option to grant
             video access. Suspend the session queue to delay session
             setup until the access request has completed.
             
             Note that audio access will be implicitly requested when we
             create an AVCaptureDeviceInput for audio during session setup.
             */
            sessionQueue.suspend()
            AVCaptureDevice.requestAccess(for: .video) { [weak self] (granted: Bool) in
                if !granted {
                    self?.setupResult = .notAuthorized
                }
                self?.sessionQueue.resume()
            }
            
        default:
            // The user has previously denied access.
            setupResult = .notAuthorized
            
            DispatchQueue.main.async { [weak self] in
                self?.alertError = AlertError(
                    title: "Camera Access",
                    message: "Campus no tiene permiso para usar la cámara, por favor cambia la configruación de privacidad",
                    primaryButtonTitle: "Configuración",
                    secondaryButtonTitle: nil,
                    primaryAction: {
                        UIApplication.shared.open(
                            URL(string: UIApplication.openSettingsURLString)!,
                            options: [:],
                            completionHandler: nil
                        )
                    },
                    secondaryAction: nil
                )
                self?.isCameraUnavailable = true
            }
        }
    }
    
    //  MARK: Session Managment
    /// Configures the camera capture session
    /// 
    /// After the session is configured, the function will automatically start the session.
    /// If everything works as expected, then the `onComplete` closure is executed and the
    /// very end.
    /// 
    /// - Parameters:
    ///   - onComplete: An optional closure to be executed when the session is configured
    ///   - additionalInput: An optional closure to be executed during the session configuration
    ///     process. This is useful for adding additional inputs to the session.
    ///     The closure should not throw any errors, should not start the session, and should
    ///     not begin or commit the configuration. If there is any configuration that needs to
    ///     modify the capture session, perform it in the session queue provided by the closure
    ///   - additionalOutput: An optional closure to be executed during the session configuration
    ///     process. This is useful for adding additional changes to the `AVCaptureVideoDataOutput` of the session.
    ///     The closure should not throw any errors, should not start the session, and should
    ///     not begin or commit the configuration. If there is any configuration that needs to
    ///     modify the capture session, perform it in the session queue provided by the closure.
    ///     If there is any configuration that needs to be performed on the output, perform it in the
    ///     `AVCaptureVideoDataOutput` provided in the closure
    ///             
    private func configureSession(
        onComplete: (() -> Void)? = nil, 
        additionalInput: ((AVCaptureSession, DispatchQueue) -> Void)? = nil,
        additionalOutput: ((AVCaptureSession, AVCaptureVideoDataOutput, DispatchQueue) -> Void)? = nil
        ) {
        if setupResult != .success {
            return
        }
        
        session.beginConfiguration()
        
        /*
         Do not create an AVCaptureMovieFileOutput when setting up the session because
         Live Photo is not supported when AVCaptureMovieFileOutput is added to the session.
         */
        session.sessionPreset = .hd1920x1080
        
        // Add video input.
        do {
            var defaultVideoDevice: AVCaptureDevice?
            
            // if let backCameraDevice = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back) {
            if let backCameraDevice: AVCaptureDevice = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: .back) {
                // If a rear dual camera is not available, default to the rear wide angle camera.
                defaultVideoDevice = backCameraDevice
            } else if let frontCameraDevice: AVCaptureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) {
                // If the rear wide angle camera isn't available, default to the front wide angle camera.
                defaultVideoDevice = frontCameraDevice
            }
            
            guard let defaultVideoDevice else {
                print("Default video device is unavailable.")
                setupResult = .configurationFailed
                session.commitConfiguration()
                return
            }
            
            let videoDeviceInput = try AVCaptureDeviceInput(device: defaultVideoDevice)
            
            if session.canAddInput(videoDeviceInput) {
                session.addInput(videoDeviceInput)
                self.videoDeviceInput = videoDeviceInput
                
            } else {
                print("Couldn't add video device input to the session.")
                setupResult = .configurationFailed
                session.commitConfiguration()
                return
            }
        } catch {
            print("Couldn't create video device input: \(error)")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        additionalInput?(session, sessionQueue)
        
        // Add the photo output.
        if session.canAddOutput(photoOutput) {
            session.addOutput(photoOutput)
            photoOutput.isHighResolutionCaptureEnabled = true
            photoOutput.maxPhotoQualityPrioritization = .quality
        } else {
            print("Could not add photo output to the session")
            setupResult = .configurationFailed
            session.commitConfiguration()
            return
        }
        
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
//        output.setSampleBufferDelegate(self, queue: sampleBufferQueue)
        additionalOutput?(session, output, sampleBufferQueue)
        session.addOutput(output)
        session.commitConfiguration()
        self.isConfigured = true
        
        self.start()
        onComplete?()
    }
    
    private func resumeInterruptedSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            /*
             The session might fail to start running, for example, if a phone or FaceTime call is still
             using audio or video. This failure is communicated by the session posting a
             runtime error notification. To avoid repeatedly failing to start the session,
             only try to restart the session in the error handler if you aren't
             trying to resume the session.
             */
            self.session.startRunning()
            self.isSessionRunning = self.session.isRunning
            if !self.session.isRunning {
                DispatchQueue.main.async { [weak self] in
                    self?.alertError = AlertError(
                        title: "Camera Error",
                        message: "Unable to resume camera",
                        primaryButtonTitle: "Accept",
                        secondaryButtonTitle: nil,
                        primaryAction: nil,
                        secondaryAction: nil
                    )
                    self?.isCameraUnavailable = true
                }
            } else {
                DispatchQueue.main.async { [weak self] in
                    self?.isCameraUnavailable = false
                }
            }
        }
    }
    
    //  MARK: Device Configuration
    
    /// - Tag: ChangeCamera
    public func changeCamera() {
        // MARK: Here disable all camera operation related buttons due to configuration is due upon and must not be interrupted
        DispatchQueue.main.async { [weak self] in
            self?.isCameraUnavailable = true
        }
        
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            let currentVideoDevice: AVCaptureDevice = self.videoDeviceInput.device
            let currentPosition: AVCaptureDevice.Position = currentVideoDevice.position
            
            let preferredPosition: AVCaptureDevice.Position
            let preferredDeviceType: AVCaptureDevice.DeviceType
            
            switch currentPosition {
            case .unspecified, .front:
                preferredPosition = .back
                preferredDeviceType = .builtInWideAngleCamera
                
            case .back:
                preferredPosition = .front
                preferredDeviceType = .builtInWideAngleCamera
                
            @unknown default:
                print("Unknown capture position. Defaulting to back, dual-camera.")
                preferredPosition = .back
                preferredDeviceType = .builtInWideAngleCamera
            }
            let devices: [AVCaptureDevice] = self.videoDeviceDiscoverySession.devices
            var newVideoDevice: AVCaptureDevice? = nil
            
            // First, seek a device with both the preferred position and device type. Otherwise, seek a device with only the preferred position.
            if let device = devices.first(where: { $0.position == preferredPosition && $0.deviceType == preferredDeviceType }) {
                newVideoDevice = device
            } else if let device = devices.first(where: { $0.position == preferredPosition }) {
                newVideoDevice = device
            }
            
            if let videoDevice = newVideoDevice {
                do {
                    let videoDeviceInput: AVCaptureDeviceInput = try AVCaptureDeviceInput(device: videoDevice)
                    
                    self.session.beginConfiguration()
                    
                    // Remove the existing device input first, because AVCaptureSession doesn't support
                    // simultaneous use of the rear and front cameras.
                    self.session.removeInput(self.videoDeviceInput)
                    
                    if self.session.canAddInput(videoDeviceInput) {
                        NotificationCenter.default.removeObserver(self, name: .AVCaptureDeviceSubjectAreaDidChange, object: currentVideoDevice)
                        NotificationCenter.default.addObserver(self, selector: #selector(self.subjectAreaDidChange), name: .AVCaptureDeviceSubjectAreaDidChange, object: videoDeviceInput.device)
                        
                        self.session.addInput(videoDeviceInput)
                        self.videoDeviceInput = videoDeviceInput
                    } else {
                        self.session.addInput(self.videoDeviceInput)
                    }
                    
                    if let connection = self.photoOutput.connection(with: .video) {
                        if connection.isVideoStabilizationSupported {
                            connection.preferredVideoStabilizationMode = .auto
                        }
                    }
                    
                    self.photoOutput.maxPhotoQualityPrioritization = .quality
                    
                    self.session.commitConfiguration()
                } catch {
                    print("Error occurred while creating video device input: \(error)")
                }
            }
        }
    }
    
    public func focus(with focusMode: AVCaptureDevice.FocusMode, exposureMode: AVCaptureDevice.ExposureMode, at devicePoint: CGPoint, monitorSubjectAreaChange: Bool) {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            guard let device = self.videoDeviceInput?.device else { return }
            do {
                try device.lockForConfiguration()
                
                /*
                 Setting (focus/exposure)PointOfInterest alone does not initiate a (focus/exposure) operation.
                 Call set(Focus/Exposure)Mode() to apply the new point of interest.
                 */
                if device.isFocusPointOfInterestSupported && device.isFocusModeSupported(focusMode) {
                    device.focusPointOfInterest = devicePoint
                    device.focusMode = focusMode
                }
                
                if device.isExposurePointOfInterestSupported && device.isExposureModeSupported(exposureMode) {
                    device.exposurePointOfInterest = devicePoint
                    device.exposureMode = exposureMode
                }
                
                device.isSubjectAreaChangeMonitoringEnabled = monitorSubjectAreaChange
                device.unlockForConfiguration()
            } catch {
                print("Could not lock device for configuration: \(error)")
            }
        }
    }
    
    public func focus(at focusPoint: CGPoint){
        let device = self.videoDeviceInput.device
        do {
            try device.lockForConfiguration()
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = focusPoint
                device.exposurePointOfInterest = focusPoint
                device.exposureMode = .continuousAutoExposure
                device.focusMode = .continuousAutoFocus
                device.unlockForConfiguration()
            }
        }
        catch {
            print(error.localizedDescription)
        }
    }
    
    @objc public func stop(completion: (() -> ())? = nil) {
        guard isSessionRunning else { return }
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.isSessionRunning {
                if self.setupResult == .success {
                    self.session.stopRunning()
//                    self.isSessionRunning = self.session.isRunning
                    self.isSessionRunning = false
                    self.removeObservers()
                    
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        self.isCameraUnavailable = true
                        completion?()
                    }
                }
            }
        }
    }
    
    @objc public func start() {
        guard !isSessionRunning else { return }
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if !self.isSessionRunning && self.isConfigured {
                switch self.setupResult {
                case .success:
                    // Only setup observers and start the session if setup succeeded.
                    self.addObservers()
                    self.session.startRunning()
//                    print("CAMERA RUNNING")
//                    self.isSessionRunning = self.session.isRunning
                    self.isSessionRunning = true
                    
                    if self.session.isRunning {
                        DispatchQueue.main.async { [weak self] in
                            self?.isCameraUnavailable = false
                        }
                    }
                    
                case .notAuthorized:
                    print("Application not authorized to use camera")
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        self.isCameraUnavailable = true
                    }
                    
                case .configurationFailed:
                    DispatchQueue.main.async { [weak self] in
                        self?.alertError = AlertError(
                            title: "Camera Error",
                            message: "Camera configuration failed. Either your device camera is not available or other application is using it",
                            primaryButtonTitle: "Accept",
                            secondaryButtonTitle: nil,
                            primaryAction: nil,
                            secondaryAction: nil
                        )
                        self?.isCameraUnavailable = true
                    }
                }
            }
        }
    }
    
    public func set(zoom: CGFloat){
        let factor = zoom < 1 ? 1 : zoom
        let device = self.videoDeviceInput.device
        
        do {
            try device.lockForConfiguration()
            device.videoZoomFactor = factor
            device.unlockForConfiguration()
        }
        catch {
            print(error.localizedDescription)
        }
    }
    
    //    MARK: Capture Photo
    
    /// - Tag: CapturePhoto
    public func capturePhoto(completionHandler: (() -> Void)? = nil) {
        /*
         Retrieve the video preview layer's video orientation on the main queue before
         entering the session queue. This to ensures that UI elements are accessed on
         the main thread and session configuration is done on the session queue.
         */
        
        if self.setupResult != .configurationFailed {
//            let videoPreviewLayerOrientation: AVCaptureVideoOrientation = .portrait
            let videoPreviewLayerOrientation: AVCaptureVideoOrientation
            switch UIDevice.current.orientation {
            case .portrait:
                videoPreviewLayerOrientation = .portrait // 1
            case .portraitUpsideDown:
                videoPreviewLayerOrientation = .portraitUpsideDown // 2
            case .landscapeLeft:
                videoPreviewLayerOrientation = .landscapeRight // 3
            case .landscapeRight:
                videoPreviewLayerOrientation = .landscapeLeft // 4
            default:
                videoPreviewLayerOrientation = .portrait // 1
            }
            
            sessionQueue.async { [weak self] in
                guard let self = self else { return }
                if let photoOutputConnection = self.photoOutput.connection(with: .video) {
                    photoOutputConnection.videoOrientation = videoPreviewLayerOrientation
                }
                var photoSettings = AVCapturePhotoSettings()
                
                // Capture HEIF photos when supported. Enable according to user settings and high-resolution photos.
                if self.photoOutput.availablePhotoCodecTypes.contains(.hevc) {
                    photoSettings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
                }
                
                if self.videoDeviceInput.device.isFlashAvailable {
                    photoSettings.flashMode = self.flashMode
                }
                
                photoSettings.isHighResolutionPhotoEnabled = true
                if !photoSettings.__availablePreviewPhotoPixelFormatTypes.isEmpty {
                    photoSettings.previewPhotoFormat = [kCVPixelBufferPixelFormatTypeKey as String: photoSettings.__availablePreviewPhotoPixelFormatTypes.first!]
                }
                
                photoSettings.photoQualityPrioritization = .speed
                
                let photoCaptureProcessor = PhotoCaptureProcessor(with: photoSettings) {
                    // Flash the screen to signal that AVCam took a photo.
                    DispatchQueue.main.async {
                        self.willCapturePhoto.toggle()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                            self.willCapturePhoto.toggle()
                        }
                        UIImpactFeedbackGenerator(style: .light).impactOccurred()
                    }
                } completionHandler: { (photoCaptureProcessor) in
                    // When the capture is complete, remove a reference to the photo capture delegate so it can be deallocated.
                    if let data = photoCaptureProcessor.photoData {
                        self.photo = Photo(originalData: data)
                    }
                    
                    self.isCameraUnavailable = true
                    
                    self.sessionQueue.async {
                        self.inProgressPhotoCaptureDelegates[photoCaptureProcessor.requestedPhotoSettings.uniqueID] = nil
                        completionHandler?()
                    }
                } photoProcessingHandler: { animate in
                    // Animates a spinner while photo is processing
                    self.shouldShowSpinner = animate
                }
                
                // The photo output holds a weak reference to the photo capture delegate and stores it in an array to maintain a strong reference.
                self.inProgressPhotoCaptureDelegates[photoCaptureProcessor.requestedPhotoSettings.uniqueID] = photoCaptureProcessor
                self.photoOutput.capturePhoto(with: photoSettings, delegate: photoCaptureProcessor)
            }
        }
    }
    
    
    //  MARK: KVO & Observers
    
    /// - Tag: ObserveInterruption
    private func addObservers() {
        let systemPressureStateObservation = observe(\.videoDeviceInput.device.systemPressureState, options: .new) { _, change in
            guard let systemPressureState = change.newValue else { return }
            self.setRecommendedFrameRateRangeForPressureState(systemPressureState: systemPressureState)
        }
        keyValueObservations.append(systemPressureStateObservation)
        
        //        NotificationCenter.default.addObserver(self, selector: #selector(self.onOrientationChange), name: UIDevice.orientationDidChangeNotification, object: nil)
        
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(subjectAreaDidChange),
                                               name: .AVCaptureDeviceSubjectAreaDidChange,
                                               object: videoDeviceInput.device)
        
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(uiRequestedNewFocusArea),
                                               name: .init(rawValue: "UserDidRequestNewFocusPoint"),
                                               object: nil)
        
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(sessionRuntimeError),
                                               name: .AVCaptureSessionRuntimeError,
                                               object: session)
        
        /*
         A session can only run when the app is full screen. It will be interrupted
         in a multi-app layout, introduced in iOS 9, see also the documentation of
         AVCaptureSessionInterruptionReason. Add observers to handle these session
         interruptions and show a preview is paused message. See the documentation
         of AVCaptureSessionWasInterruptedNotification for other interruption reasons.
         */
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(sessionWasInterrupted),
                                               name: .AVCaptureSessionWasInterrupted,
                                               object: session)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(sessionInterruptionEnded),
                                               name: .AVCaptureSessionInterruptionEnded,
                                               object: session)
    }
    
    private func removeObservers() {
        NotificationCenter.default.removeObserver(self)
        
        for keyValueObservation in keyValueObservations {
            keyValueObservation.invalidate()
        }
        keyValueObservations.removeAll()
    }
    
    @objc private func uiRequestedNewFocusArea(notification: NSNotification) {
        guard let userInfo = notification.userInfo as? [String: Any], let devicePoint = userInfo["devicePoint"] as? CGPoint else { return }
        self.focus(at: devicePoint)
    }
    
    @objc
    private func subjectAreaDidChange(notification: NSNotification) {
        let devicePoint = CGPoint(x: 0.5, y: 0.5)
        focus(with: .continuousAutoFocus, exposureMode: .continuousAutoExposure, at: devicePoint, monitorSubjectAreaChange: false)
    }
    
    /// - Tag: HandleRuntimeError
    @objc
    private func sessionRuntimeError(notification: NSNotification) {
        guard let error = notification.userInfo?[AVCaptureSessionErrorKey] as? AVError else { return }
        
        print("Capture session runtime error: \(error)")
        // If media services were reset, and the last start succeeded, restart the session.
        if error.code == .mediaServicesWereReset {
            sessionQueue.async {
                if self.isSessionRunning {
                    self.session.startRunning()
                    self.isSessionRunning = self.session.isRunning
                } else {
                    DispatchQueue.main.async {
                        //                        self.resumeButton.isHidden = false
                    }
                }
            }
        } else {
            //            resumeButton.isHidden = false
        }
    }
    
    /// - Tag: HandleSystemPressure
    private func setRecommendedFrameRateRangeForPressureState(systemPressureState: AVCaptureDevice.SystemPressureState) {
        /*
         The frame rates used here are only for demonstration purposes.
         Your frame rate throttling may be different depending on your app's camera configuration.
         */
        let pressureLevel: AVCaptureDevice.SystemPressureState.Level = systemPressureState.level
        if pressureLevel == .serious || pressureLevel == .critical {
            do {
                try self.videoDeviceInput.device.lockForConfiguration()
                print("WARNING: Reached elevated system pressure level: \(pressureLevel). Throttling frame rate.")
                self.videoDeviceInput.device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 20)
                self.videoDeviceInput.device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 15)
                self.videoDeviceInput.device.unlockForConfiguration()
            } catch {
                print("Could not lock device for configuration: \(error)")
            }
        } else if pressureLevel == .shutdown {
            print("Session stopped running due to shutdown system pressure level.")
        }
    }
    
    /// - Tag: HandleInterruption
    @objc
    private func sessionWasInterrupted(notification: NSNotification) {
        /*
         In some scenarios you want to enable the user to resume the session.
         For example, if music playback is initiated from Control Center while
         using Campus, then the user can let Campus resume
         the session running, which will stop music playback. Note that stopping
         music playback in Control Center will not automatically resume the session.
         Also note that it's not always possible to resume, see `resumeInterruptedSession(_:)`.
         */
        DispatchQueue.main.async {
            self.isCameraUnavailable = true
        }
        
        if let userInfoValue = notification.userInfo?[AVCaptureSessionInterruptionReasonKey] as AnyObject?,
           let reasonIntegerValue = userInfoValue.integerValue,
           let reason = AVCaptureSession.InterruptionReason(rawValue: reasonIntegerValue) {
            print("Capture session was interrupted with reason \(reason)")
            
            if reason == .audioDeviceInUseByAnotherClient || reason == .videoDeviceInUseByAnotherClient {
                print("Session stopped running due to video devies in use by another client.")
            } else if reason == .videoDeviceNotAvailableWithMultipleForegroundApps {
                // Fade-in a label to inform the user that the camera is unavailable.
                print("Session stopped running due to video devies is not available with multiple foreground apps.")
            } else if reason == .videoDeviceNotAvailableDueToSystemPressure {
                print("Session stopped running due to shutdown system pressure level.")
            }
        }
    }
    
    @objc
    private func sessionInterruptionEnded(notification: NSNotification) {
        print("Capture session interruption ended")
        DispatchQueue.main.async {
            self.isCameraUnavailable = false
        }
    }
}

//extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
//
//    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
//        guard let model = try? VNCoreMLModel(for: YOLOv4(configuration: .init()).model) else { return }
//        let request = VNCoreMLRequest(model: model, completionHandler: visionRequestDidComplete)
//        let curDeviceOrientation = UIDevice.current.orientation
//        let exifOrientation: CGImagePropertyOrientation
//
//        switch curDeviceOrientation {
//        case UIDeviceOrientation.portraitUpsideDown:  // Device oriented vertically, home button on the top
//            exifOrientation = .left
//        case UIDeviceOrientation.landscapeLeft:       // Device oriented horizontally, home button on the right
//            exifOrientation = .upMirrored
//        case UIDeviceOrientation.landscapeRight:      // Device oriented horizontally, home button on the left
//            exifOrientation = .down
//        case UIDeviceOrientation.portrait:            // Device oriented vertically, home button on the bottom
//            exifOrientation = .up
//        default:
//            exifOrientation = .up
//        }
//
//        request.imageCropAndScaleOption = .scaleFill // filled to 416,416
//        DispatchQueue.global().async {
//            let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: exifOrientation)
//            do {
//                try handler.perform([request])
//            } catch {
//                print(error)
//            }
//        }
//    }
//
//}
