//
//  CameraView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/11/2022.
//

import SwiftUI

struct CameraView: View {
    @ObservedObject var cvm: CameraViewModel
    let screenSize: CGSize
    
    @State private var isBarCodeIndicatorViewAppear: Bool = false
        
    var body: some View {
        ZStack {
            if cvm.captureSource != nil {
                DisplayedImageView(
                    image: cvm.displayedImage ?? UIImage(),
                    isScaleToFill: $cvm.isScaleToFill,
                    screenSize: screenSize,
                    barcode: cvm.scanBarcode.barcode,
                    normalizedBarcodeBBox: cvm.scanBarcode.normalizedBbox,
                    isNutritionTableDetected: cvm.ntDetection.boundingBox != nil,
                    isLoadingInputProductDetailView: cvm.isLoadingInputProductDetailView,
                    didAnalysisButtonCliced: cvm.didAnalysisButtonCliced,
                    convertNormalizedBBoxToRectInSpecificView: cvm.scanBarcode.getConvertedRect,
                    convertImage: cvm.scanBarcode.convertImage,
                    transform: cvm.scanBarcode.transform,
                    isYoloInitializing: cvm.ntDetection.model == nil,
                    onYoloFinishedInitializing: cvm.onYoloFinishedInitializing
                )
            } else {
                CameraPreview(session: cvm.cameraService.session)
                    .overlay(alignment: .topLeading) {
                        if let bbox: CGRect = cvm.scanBarcode.boundingBox {
                            BarCodeIndicatorView(
                                barcode: cvm.scanBarcode.barcode,
                                width: bbox.width,
                                height: bbox.height,
                                offset: bbox.origin,
                                isAppeared: $isBarCodeIndicatorViewAppear
                            )
                            .onAppear { isBarCodeIndicatorViewAppear = true }
                            .onDisappear { isBarCodeIndicatorViewAppear = false }
                            .animation(.easeIn, value: cvm.scanBarcode.boundingBox)
                        }
                    }
                    .onTapGesture(perform: cvm.onCameraPreviewTap)
                    .overlay(alignment: .top) {
                        BarcodeHeader(
                            barcode: $cvm.scanBarcode.barcode,
                            onXmarkButPressed: cvm.onXmarkButPressed,
                            onSearchButtonPressed: cvm.onSearchButtonPressed
                        )
                    }
            }
            Color.black
                .opacity(cvm.cameraService.willCapturePhoto ? 1 : 0)
        }
        .frame(width: screenSize.width, height: screenSize.height)
        .clipped()
        .edgesIgnoringSafeArea(.all)
        .sheet(
            isPresented: $cvm.pickerService.showImagePicker,
            onDismiss: cvm.captureGalleryImage
        ) {
            ImagePicker(
                showImagePicker: $cvm.pickerService.showImagePicker,
                image: $cvm.pickerService.image,
                camera: false
            )
        }
        .sheet(
            isPresented: Binding<Bool>(
                get: { cvm.detail != nil },
                set: { (isPresented: Bool) in
                    if !isPresented {
                        cvm.detail = nil
                    }
                }
            )
        ) {
            /// Note:
            /// The `InputProductDetailView` itself does not contain `NavigationStack`
            /// therefore, it is the parent view's (this view) responsibility to
            /// embed `InputProductDetailView` to `NavigationStack` so that the
            /// navigation bar and large navigation title could be displayed properly
            NavigationStack {
                InputProductDetailView(detail: cvm.detail!)
            }
        }
        .sheet(
            isPresented: Binding<Bool>(
                get: { cvm.matchedBarcodeProduct != nil },
                set: { (isPresented: Bool) in
                    if !isPresented {
                        cvm.matchedBarcodeProduct = nil
                    }
                }
            )
        ) {
            NavigationStack {
                InputProductDetailView(detail: cvm.matchedBarcodeProduct!)
            }
        }
        .alert(
            "Error",
            isPresented: Binding<Bool>(
                get: { cvm.scanBarcode.errorMessage != nil },
                set: { _ in cvm.scanBarcode.errorMessage = nil }
            )
        ) {
            Button(role: .cancel) {
                
            } label: {
                Text("OK")
            }
        } message: {
            Text(cvm.scanBarcode.errorMessage ?? "")
        }

    }
}

fileprivate struct BarcodeHeader: View {
    @Environment(\.safeAreaInsets) var safeArea
    @Binding var barcode: String
    let onXmarkButPressed: () -> Void
    let onSearchButtonPressed: () -> Void
    var body: some View {
        VStack {
            HStack {
                Image(systemName: "barcode.viewfinder")
                    .foregroundColor(.systemBlue)
                TextField(text: Binding<String>(
                    get: { barcode },
                    set: { (s: String, _) in
                        withAnimation(.spring()) {
                            barcode = s
                        }
                    }
                ))
                .keyboardType(.numberPad)
                .overlay(alignment: .leading) {
                    if barcode.isEmpty {
                        Text("Product Barcode")
                            .allowsHitTesting(false)
                    }
                }
                .overlay(alignment: .trailing) {
                    Button(action: onXmarkButPressed) {
                        Circle()
                            .fill(.ultraThickMaterial)
                            .overlay {
                                Image(systemName: "xmark")
                                    .resizable()
                                    .scaledToFit()
                                    .padding(8)
                                    .foregroundColor(.primary)
                            }
                    }
                    .hoverEffect()
                }
            }
            .padding()
            .background {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(.thinMaterial)
            }
            .zIndex(1)
            if !barcode.isEmpty {
                Button(action: onSearchButtonPressed) {
                    Text("Search")
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(.systemBlue)
                        .cornerRadius(20, style: .continuous)
                }
                .opacity(barcode.isEmpty ? 0 : 1)
                .transition(
                    .asymmetric(
                        insertion: .move(edge: .top),
                        removal: .identity)
                )
                .zIndex(0)
                .hoverEffect()
            }
        }
        .productFont(.bold, relativeTo: .title2)
        .foregroundColor(.primary)
        .padding(.horizontal)
        .padding(.top, safeArea.top)
        .padding(.vertical)
        .background {
            Rectangle()
                .fill(.ultraThinMaterial)
        }
        .transition(.move(edge: .top))
    }
}

fileprivate struct DisplayedImageView: View {
    
    let image: UIImage
    @Binding var isScaleToFill: Bool
    let screenSize: CGSize
    let barcode: String
    let normalizedBarcodeBBox: CGRect?
    let isNutritionTableDetected: Bool
    let isLoadingInputProductDetailView: Bool
    let didAnalysisButtonCliced: (() -> Void)?
    let convertNormalizedBBoxToRectInSpecificView: (CGRect, CGSize, CGSize, ContentMode) -> CGRect
    let convertImage: (CGSize, CGSize, ContentMode) -> CGRect
    let transform: (CGRect, CGRect) -> CGRect
    let isYoloInitializing: Bool
    let onYoloFinishedInitializing: (() -> Void)
    
    @State private var isBarCodeIndicatorViewAppear: Bool = false
    
    var body: some View {
        ZoomableScrollView(
            isScaleToFill: $isScaleToFill,
            size: screenSize
        ) {
            Image(uiImage: image)
                .resizable()
                .aspectRatio(contentMode: isScaleToFill ? .fill : .fit)
                .overlay(alignment: .topLeading) {
                    GeometryReader { (proxy: GeometryProxy) in
                        /// Note:
                        /// This `containerSize` will keep update its value when user pinch to zoom the image
                        /// in the scrollview, then the `rectOfImage` will be called and keep up-to-dated,
                        /// therefore it will result in the `BarCodeIndicatorView` well-positioned w.r.t.
                        /// user zooming direction and position
                        ///
                        /// Btw, this `containerSize` is not the same as screen size, because for example, when the
                        /// content mode is `scaleToFill`, there will be area in iamge overflowed outside the screen view,
                        /// so we need this `containerSize` to keep track of the entire image size (including the overflowed area)
                        let containerSize: CGSize = proxy.frame(in: .local).size
                        
                        if let normalizedBarcodeBBox {
                            let rectOfImage: CGRect = convertImage(image.size, containerSize, isScaleToFill ? .fill : .fit)
                            /// un-normalized bounding box, given the real image frame (which contains
                            /// the real image size to be scaled, and the position to be offseted) placed on container
                            let convertedRect: CGRect = transform(normalizedBarcodeBBox, rectOfImage)
                            BarCodeIndicatorView(
                                barcode: barcode,
                                width: convertedRect.width,
                                height: convertedRect.height,
                                offset: convertedRect.origin,
                                isAppeared: $isBarCodeIndicatorViewAppear
                            )
                        }
                    }
                }
                .frame(width: screenSize.width, height: screenSize.height)
        }
        .overlay(alignment: .top) {
            Header(
                barcode: barcode,
                isNutritionTableDetected: isNutritionTableDetected,
                isLoadingInputProductDetailView: isLoadingInputProductDetailView,
                didAnalysisButtonCliced: didAnalysisButtonCliced,
                isYoloInitializing: isYoloInitializing,
                onYoloFinishedInitializing: onYoloFinishedInitializing
            )
        }
        .onAppear {
            isBarCodeIndicatorViewAppear = true
        }
    }
    
    private func toBinding<T: View>(v: T) -> Binding<T> {
        Binding<T>(
            get: { v },
            set: { _ in }
        )
    }
    
    fileprivate struct Header: View {
        @Environment(\.safeAreaInsets) var safeArea
        let barcode: String
        let isNutritionTableDetected: Bool
        let isLoadingInputProductDetailView: Bool
        let didAnalysisButtonCliced: (() -> Void)?
        let isYoloInitializing: Bool
        let onYoloFinishedInitializing: (() -> Void)
        var body: some View {
            VStack(spacing: 16) {
                Button {
                    /// To avoid sending multiple request to server at the same time
                    if !isLoadingInputProductDetailView {
                        didAnalysisButtonCliced?()
                    }
                } label: {
                    Group {
                        if isLoadingInputProductDetailView {
                            ProgressView()
                        } else {
                            VStack {
                                Image(systemName: "tablecells.badge.ellipsis")
                                Text("Analysis")
                                    .productFont(.bold, relativeTo: .title3)
                            }
                            .foregroundStyle(.secondary)
                        }
                    }
                    .frame(height: 50)
                    .frame(maxWidth: .infinity)
                }
                .frame(maxWidth: .infinity)
                .hoverEffect()
                .buttonStyle(.plain)
                Rectangle()
                    .frame(height: 1)
                    .frame(maxWidth: .infinity)
                    .foregroundStyle(.secondary)
                HStack {
                    VStack(alignment: .trailing, spacing: 8) {
                        Text("Barcode: ")
                        Text("Nutrition Table: ")
                    }
                    .productFont(.bold, relativeTo: .body)
                    .foregroundStyle(.primary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    VStack(alignment: .leading, spacing: 8) {
                        Text(barcode.isEmpty ? "Not Found" : barcode)
                        if isYoloInitializing {
                            ProgressView()
                        } else {
                            Text(isNutritionTableDetected ? "Detected" : "Not Found")
                        }
                    }
                    .productFont(.regular, relativeTo: .body)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(.top, safeArea.top)
            .padding()
            .background(.thinMaterial, in: Rectangle())
            .transition(.move(edge: .top))
            .onChange(of: isYoloInitializing) { (newValue: Bool) in
                if !newValue {
                    onYoloFinishedInitializing()
                }
            }
        }
    }
}

struct CameraView_Previews: PreviewProvider {
    @StateObject static var mvm = MainViewModel()
    static var previews: some View {
        ContentView()
        .environmentObject(mvm)
    }
}
