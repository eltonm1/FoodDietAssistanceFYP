//
//  PhotoDetailView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 14/2/2023.
//

import SwiftUI
import PDFKit

struct PhotoDetailView: UIViewRepresentable {
    let image: UIImage
    let size: CGSize

    func makeUIView(context: Context) -> PDFView {
        let view = PDFView()
        view.frame.size = size
        view.document = PDFDocument()
        guard let page = PDFPage(image: image) else { return view }
        view.document?.insert(page, at: 0)
        view.autoScales = true
        // only use PostScript names when calling this API.
        view.displayMode = .singlePageContinuous
        view.displayDirection = .vertical
        view.minScaleFactor = view.scaleFactorForSizeToFit
        return view
    }

    func updateUIView(_ uiView: PDFView, context: Context) {
        // empty
    }
}
