//
//  ZoomableScrollView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 14/2/2023.
//

import Foundation
import SwiftUI
import UIKit

struct ZoomableScrollView<Content: View>: UIViewRepresentable {
//struct ZoomableScrollView: UIViewRepresentable {
//    let image: () -> UIImage?
//    @Binding var isScaleToFill: Bool
//    let size: CGSize
//
//    func makeCoordinator() -> Coordinator {
//        Coordinator(self)
//    }
//
//    func makeUIView(context: Context) -> UIScrollView {
//        let scrollView = UIScrollView()
//        let img: UIImage? = image()
//        let imageView = UIImageView(image: img)
//        imageView.frame.size = img?.size ?? .zero
//        imageView.contentMode = isScaleToFill ? .scaleAspectFill : .scaleAspectFit
//        scrollView.delegate = context.coordinator
//        scrollView.frame.size = size
//        scrollView.alwaysBounceVertical = false
//        scrollView.alwaysBounceHorizontal = false
//        scrollView.showsVerticalScrollIndicator = false
//        scrollView.showsHorizontalScrollIndicator = false
//        scrollView.flashScrollIndicators()
//        scrollView.contentSize = size
//        scrollView.isScrollEnabled = false
//
//        scrollView.addSubview(imageView)
//        scrollView.minimumZoomScale = 1.0
//        scrollView.maximumZoomScale = 6.0
//        animateContentMode(imageView: imageView)
//        return scrollView
//    }
//
//    func updateUIView(_ uiView: UIScrollView, context: Context) {
//        let imageView = uiView.subviews.first as? UIImageView
//        imageView?.image = image()
//        uiView.setZoomScale(1.0, animated: true)
//        animateContentMode(imageView: imageView!)
//    }
//
//    private func animateContentMode(imageView: UIImageView) {
//        // animate by frame, and set contentMode after animation
//        guard let img: UIImage = imageView.image else { return }
//        let imageWidthToHeightRatio: CGFloat = img.size.width / img.size.height
//        if !isScaleToFill { // scale to fit
//            let newImageSize: CGSize = .init(width: size.width, height: size.width / imageWidthToHeightRatio)
//            let newFrame: CGRect = .init(
//                x: 0,
//                y: (size.height - newImageSize.height) / 2,
//                width: newImageSize.width,
//                height: newImageSize.height
//            )
//            UIView.animate(withDuration: 0.3) {
//                imageView.frame = newFrame
//            } completion: { _ in
//                imageView.contentMode = .scaleAspectFit
//            }
//        } else { // scale to fill
//            let newImageSize: CGSize = .init(width: size.height * imageWidthToHeightRatio, height: size.height)
//            let newFrame: CGRect = .init(
//                x: (size.width - newImageSize.width) / 2,
//                y: 0,
//                width: newImageSize.width,
//                height: newImageSize.height
//            )
//            UIView.animate(withDuration: 0.3) {
//                imageView.frame = newFrame
//            } completion: { _ in
//                imageView.contentMode = .scaleAspectFill
//            }
//        }
//    }
    let view: () -> Content
    @Binding var isScaleToFill: Bool
    let size: CGSize
    
    init(
        isScaleToFill: Binding<Bool>,
        size: CGSize,
        view: @escaping () -> Content
    ) {
        self.view = view
        self._isScaleToFill = isScaleToFill
        self.size = size
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> UIScrollView {
        let scrollView = UIScrollView()
        let uiView: UIView = UIHostingController(rootView: view()).view
        uiView.frame = .init(origin: .zero, size: size)
        scrollView.delegate = context.coordinator
        scrollView.frame.size = size
        scrollView.alwaysBounceVertical = false
        scrollView.alwaysBounceHorizontal = false
        scrollView.showsVerticalScrollIndicator = false
        scrollView.showsHorizontalScrollIndicator = false
        scrollView.flashScrollIndicators()
        scrollView.contentSize = size
        scrollView.isScrollEnabled = false

        scrollView.addSubview(uiView)
        scrollView.minimumZoomScale = 1.0
        scrollView.maximumZoomScale = 6.0
        return scrollView
    }

    func updateUIView(_ uiView: UIScrollView, context: Context) {
        uiView.subviews.forEach { $0.removeFromSuperview() }
        let v: UIView = UIHostingController(rootView: view()).view
        v.frame = .init(origin: .zero, size: size)
        uiView.addSubview(v)
        uiView.setZoomScale(1.0, animated: true)
    }
    
    class Coordinator: NSObject, UIScrollViewDelegate {
        var parent: ZoomableScrollView

        init(_ parent: ZoomableScrollView) {
            self.parent = parent
        }

        func viewForZooming(in scrollView: UIScrollView) -> UIView? {
            scrollView.subviews.first
        }
    }
}
