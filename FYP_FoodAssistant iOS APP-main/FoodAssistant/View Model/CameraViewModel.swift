//
//  CameraViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import Foundation
import Combine
import UIKit
import SwiftUI
import Vision

class CameraViewModel: ObservableObject {
    
    // MARK: - Services
    @Published var cameraService: CameraService
    @Published var pickerService = ImagePickerService()
    @Published var ntDetection = NutritionTableDetectionService()
    @Published var scanBarcode: ScanBarcodeService
    @Published var foodDataService: FoodProductDataService
    // MARK: - Photo Capture
    @Published var captureSource: CaptureSource?
    // MARK: - Camera Bottom Bar
    @Published var isScaleToFill: Bool = true
    // MARK: - Barcode Header
    @Published var isEditing = false
    // MARK: - Displayed Image View Header Buttons
    /// When set, the `CameraView` will automatically present the `InputProductDetailView` sheet view,
    /// and reset to nil when dismissed. Setting this variable to nil will also dismiss the `InputProductDetailView` as well.
    @Published var detail: InputProductDetailView.Detail? {
        didSet {
            if detail != nil {
                isLoadingInputProductDetailView = false
            }
        }
    }
    /// When user clicks the search button on barcode header, and there is a matching product whose barcode is the same
    /// as user inputted on the header, then this value will be set and a `InputProductDetailView` sheet view
    /// will be presented accordingly. Setting this value to nil will dismiss the sheet
    @Published var matchedBarcodeProduct: InputProductDetailView.Detail?
    /// It takes time for server to perform second stage. During the mean time, this variable will control whether or not
    /// to display the activity indicator view on button to let user know that the app is waiting for server response, not
    /// freezing
    @Published var isLoadingInputProductDetailView: Bool = false
    // MARK: Cancellables to Store All Subscribers
    var anyCancellables = Set<AnyCancellable>()
    
    // MARK: - Init
    init(
        cameraService: CameraService,
        foodDataService: FoodProductDataService
    ) {
        self._cameraService = Published(wrappedValue: cameraService)
        self._scanBarcode = Published(wrappedValue: ScanBarcodeService(cameraService: cameraService))
        self._foodDataService = Published(wrappedValue: foodDataService)
        
        scanBarcode.onAppear()
        
        cameraService.objectWillChange.sink { [weak self] (_) in
            self?.objectWillChange.send()
        }
        .store(in: &anyCancellables)

        pickerService.objectWillChange.sink { [weak self] (_) in
            self?.objectWillChange.send()
        }
        .store(in: &anyCancellables)
        
        ntDetection.objectWillChange.sink { [weak self] (_) in
            self?.objectWillChange.send()
        }
        .store(in: &anyCancellables)
        
        scanBarcode.objectWillChange.sink { [weak self] (_) in
            self?.objectWillChange.send()
        }
        .store(in: &anyCancellables)
        
        foodDataService.objectWillChange.sink { [weak self] (_) in
            self?.objectWillChange.send()
        }
        .store(in: &anyCancellables)
    }
    
    // MARK: - Private Logic
    private func captureCameraPhoto() {
        cameraService.capturePhoto { [weak self] in
            guard let self = self,
                  let photo: Photo = self.cameraService.photo,
                  let image: UIImage = photo.image,
                  let resizedImage: UIImage = photo.resizedImage else { return }
            self.cameraService.stop { [weak self] in
                guard let self = self else { return }
                self.ntDetection.detectNuritionTable(image: resizedImage)
                self.scanBarcode.detectBarcode(from: image, on: .byCamera)
                
                Task {
                    await MainActor.run { [weak self] in
                        self?.captureSource = .byCamera
                    }
                }
            }
        }
    }
    
    // MARK: - View Logic
    var displayedImage: UIImage? {
        if let captureSource {
            switch captureSource {
            case .byCamera:
                return cameraService.photo?.image
            case .byImagePicker:
                return pickerService.photo?.image
            }
        } else {
            return nil
        }
    }

    var resacledImageSize: CGSize {
        switch captureSource {
        case .none: return .zero
        case .some(let wrapped):
            switch wrapped {
            case .byCamera:
                return cameraService.photo?.rescaledImage?.size ?? .zero
            case .byImagePicker:
                return pickerService.photo?.rescaledImage?.size ?? .zero
            }
        }
    }
    
    var displayedPhoto: Photo? {
        if let captureSource {
            switch captureSource {
            case .byCamera:
                return cameraService.photo
            case .byImagePicker:
                return pickerService.photo
            }
        } else {
            return nil
        }
    }
    
    /// Show the `InputProductDetailView` via sheet
    func didAnalysisButtonCliced() {
        /// Don't do anything if user didn't input/scan the barcode or the model is still initializing
        if scanBarcode.barcode.isEmpty || ntDetection.model == nil {
            return
        }
        
        /// If there is no croppedTable, then we can not perform the second stage, so simply return
        /// This croppedTable image is only available only after the model is initialized, so it is
        /// basically adding another check to the previous if-statement
        guard let croppedTable: UIImage = self.ntDetection.cropTable() else {
            return
        }
        
        isLoadingInputProductDetailView = true
        
        Task {
            do {
                var product: Product = try await getProductIfItExistOnServer()
                do {
//                    let nutritionInfo: NutritionInformation = try await performSecondStage(on: croppedTable)
//                    product.nutrition = nutritionInfo
                    await MainActor.run { [product] in
                        self.detail = InputProductDetailView.Detail(
                            product: product,
                            boundingBox: ntDetection.boundingBox,
                            nutritionTablePhoto: displayedPhoto,
                            /// This product is on the server record, so its inforamtion cannot be editable
                            editable: false,
                            onUpload: foodDataService.putData
                        )
                    }
                } catch {
                    print("Couldn't post image: \(error)")
                    isLoadingInputProductDetailView = false
                }
            } catch {
                do {
                    let nutritionInfo: NutritionInformation = try await performSecondStage(on: croppedTable)
                    let product = Product(barcode: self.scanBarcode.barcode, nutrition: nutritionInfo)
                    print(nutritionInfo)
                    await MainActor.run {
                        self.detail = InputProductDetailView.Detail(
                            product: product,
                            boundingBox: ntDetection.boundingBox,
                            nutritionTablePhoto: displayedPhoto,
                            /// This product is not on the server record, so its inforamtion is editable
                            editable: true,
                            onUpload: foodDataService.putData
                        )
                    }
                } catch {
                    print("Couldn't post image: \(error)")
                    isLoadingInputProductDetailView = false
                }
            }
        }
    }
    
    /// Perform API get call with the product barcode cached from `ScanBarcodeService`,
    ///
    /// - Returns: An optional Product object, indicating if the server contains the product or not
    func getProductIfItExistOnServer() async throws -> Product {
        try await AppState.shared.dataService.get(type: Product.self,path: "/api/foodproducts/" + scanBarcode.barcode)
    }
    
    /// Perform API post call with the cropped nutrition table image
    /// 
    /// - Parameter croppedTable: an UIImage object of the cropped nutrition table, this
    /// image should be returned from the cropTable function from `NutritionTableDetectionService`
    /// - Returns: a NutritionInformation object
    func performSecondStage(on croppedTable: UIImage) async throws -> NutritionInformation {
        let secstageHost: String = "20.187.76.166"
        let secstagePort: Int = 9999
        let secstagePath: String = "/predict"
        return try await AppState
            .shared
            .dataService
            .post(
                type: NutritionInformation.self,
                image: croppedTable,
                host: secstageHost,
                port: secstagePort,
                path: secstagePath
            )
    }
    
    /// Dismiss keyboard when user taps on any area on the `CameraView`
    func onCameraPreviewTap() {
        UIApplication
            .shared
            .sendAction(
                #selector(UIResponder.resignFirstResponder),
                to: nil,
                from: nil,
                for: nil
            )
    }
    
    /// Clear the text in the barcode search textfield
    func onXmarkButPressed() {
        withAnimation(.spring()) {
            scanBarcode.barcode = ""
        }
    }
    
    /// Action to be triggered when user taps on the capture button at the middle of the `CameraBottomBar`
    ///
    /// When called, when there is currently no captured photo (either from camear or image picker),
    /// the function to capture the image from camera, otherwise it will reset the captured image and
    /// start the camera session again
    func onSnapButtonTapped() {
        if let captureSource {
            self.captureSource = nil
            switch captureSource {
            case .byImagePicker:
                pickerService.photo = nil
            case .byCamera:
                cameraService.photo = nil
            }
            cameraService.start()
            scanBarcode.barcode = ""
            scanBarcode.boundingBox = nil
            scanBarcode.normalizedBbox = nil
        } else {
            captureCameraPhoto()
        }
    }
    
    /// Action to be triggered when user has tapped the trailing button on the `CameraBottomBar`
    ///
    /// When called, when there is currently no captured photo (either from camear or image picker),
    /// the function will toggle the `isScaleToFill` published property, otherwise the function
    /// will perform call to change the camera (between front and back camera)
    func onTrailingButtonTapped() {
        if captureSource != nil {
            withAnimation(.easeInOut) {
                isScaleToFill.toggle()
            }
        } else {
            cameraService.changeCamera()
        }
    }
    
    /// Capture Gallert Image
    ///
    /// Call this function when user has tapped/selected an image from photo library
    /// The function will automatically look for the image from the `ImagePickerService`,
    /// so there is no need to pass the image to the function
    ///
    /// After getting the image, the function will detect barcode and nutrition table from the image
    func captureGalleryImage() {
        guard let photo: Photo = pickerService.photo,
              let resizedImage: UIImage = photo.resizedImage,
              let image: UIImage = photo.image
        else { return }
        cameraService.stop()
        captureSource = .byImagePicker
        Task { [weak self] in
            self?.ntDetection.detectNuritionTable(image: resizedImage)
            self?.scanBarcode.detectBarcode(from: image, on: .byImagePicker)
        }
    }
    
    /// Things to do after YOLOv4 model is initialized in the background thread
    ///
    /// User could have already taken an image and is waiting for the model to be
    /// initialized so that they can detect the nutrition table. This function will be called
    /// to notify that YOLOv4 is initialized
    func onYoloFinishedInitializing() {
        guard let captureSource else { return }
        let photo: Photo?
        switch captureSource {
        case .byCamera:
            photo = cameraService.photo
        case .byImagePicker:
            photo = pickerService.photo
        }
        if let photo: Photo,
           let resizedImage: UIImage = photo.resizedImage {
            ntDetection.detectNuritionTable(image: resizedImage)
        }
    }
    
    /// Present `InputProductDetailView` sheet after user has clicked on the search button on the barcode header
    /// and there is matching product whose barcode is the same as that on header
    func onSearchButtonPressed() {
        if scanBarcode.barcode.isEmpty { return }
        guard let product: Product = foodDataService.products.first(where: { (p: Product) in
            p.barcode == scanBarcode.barcode
        }) else { return }
        self.matchedBarcodeProduct = .init(product: product, editable: false)
    }
    
    /// Returns similiar products based on scanned/manually-typped barcode
    /// after user has clicked on the search button on the barcode header
    var similarProducts: [Product] {
        foodDataService.searchSimilarProducts(by: scanBarcode.barcode)
    }
}
