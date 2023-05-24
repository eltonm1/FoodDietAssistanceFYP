//
//  ChatPreview.swift
//  CirGo
//
//  Created by 徐煒立 on 27/8/2021.
//

import SwiftUI
import SwiftUIX
import UIKit

struct PreviewContextViewModifier<Preview: View, Destination: View, NavigationValue: Hashable>: ViewModifier {
    
    @State private var isActive: Bool = false
    private let previewContent: Preview?
    private let destination: Destination?
    private let navigationValue: NavigationValue?
    private let preferredContentSize: CGSize?
    private let actions: [UIAction]
    private let presentAsSheet: Bool
    private let didCommitView: (() -> Void)?
    
    // Both `preview` and `destination` are different
    init(
        destination: Destination,
        preview: Preview,
        navigationValue: NavigationValue? = nil,
        preferredContentSize: CGSize? = nil,
        presentAsSheet: Bool = false,
        didCommitView: (() -> Void)? = nil,
        @ButtonBuilder actions: () -> [PreviewContextAction] = { [] }
    ) {
        self.destination = destination
        self.previewContent = preview
        self.navigationValue = navigationValue
        self.preferredContentSize = preferredContentSize
        self.presentAsSheet = presentAsSheet
        self.didCommitView = didCommitView
        self.actions = actions().map(\.uiAction)
    }
    
    // Both `preview` and `destination` are the same
    init(
        destination: Destination,
        navigationValue: NavigationValue? = nil,
        preferredContentSize: CGSize? = nil,
        presentAsSheet: Bool = false,
        didCommitView: (() -> Void)? = nil,
        @ButtonBuilder actions: () -> [PreviewContextAction] = { [] }
    ) {
        self.destination = destination
        self.previewContent = nil
        self.navigationValue = navigationValue
        self.preferredContentSize = preferredContentSize
        self.presentAsSheet = presentAsSheet
        self.didCommitView = didCommitView
        self.actions = actions().map(\.uiAction)
    }
    
    // Only `preivew`, no destination
    init(
        preview: Preview,
        navigationValue: NavigationValue? = nil,
        preferredContentSize: CGSize? = nil,
        presentAsSheet: Bool = false,
        didCommitView: (() -> Void)? = nil,
        @ButtonBuilder actions: () -> [PreviewContextAction] = { [] }
    ) {
        self.destination = nil
        self.previewContent = preview
        self.navigationValue = navigationValue
        self.preferredContentSize = preferredContentSize
        self.presentAsSheet = presentAsSheet
        self.didCommitView = didCommitView
        self.actions = actions().map(\.uiAction)
    }

    @ViewBuilder
    public func body(content: Content) -> some View {
        ZStack {
            if presentAsSheet {
                content
                    .sheet(isPresented: $isActive) {
                        destination
                    }
            } else {
                NavigationLink(value: navigationValue) {
                    content
                }
                .navigationDestination(isPresented: $isActive) {
                    destination
                }
            }
        }
        .overlay {
            PreviewContextView(
                preview: preview,
                preferredContentSize: preferredContentSize,
                actions: actions,
                isPreviewOnly: destination == nil,
                didCommitView: didCommitView,
                isActive: $isActive
            )
            .opacity(0.05)
        }
    }
    
    @ViewBuilder
    private var preview: some View {
        if let preview = previewContent {
            preview
        } else {
            destination
        }
    }
}

struct PreviewContextView<Preview: View>: UIViewRepresentable {

    let preview: Preview?
    let preferredContentSize: CGSize?
    let actions: [UIAction]
    let isPreviewOnly: Bool
    let didCommitView: (() -> Void)?
    @Binding var isActive: Bool
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        view.addInteraction(
            UIContextMenuInteraction(
                delegate: context.coordinator
            )
        )
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIContextMenuInteractionDelegate {
        
        private let view: PreviewContextView<Preview>
        
        init(_ view: PreviewContextView<Preview>) {
            self.view = view
        }
        
        func contextMenuInteraction(
            _ interaction: UIContextMenuInteraction,
            configurationForMenuAtLocation location: CGPoint
        ) -> UIContextMenuConfiguration? {
            UIContextMenuConfiguration(
                identifier: nil,
                previewProvider: {
                    let hostingController = UIHostingController(rootView: self.view.preview)
                    
                    if let preferredContentSize = self.view.preferredContentSize {
                        hostingController.preferredContentSize = preferredContentSize
                    }
                    
                    return hostingController
                }, actionProvider: { _ in
                    UIMenu(title: "", children: self.view.actions)
                }
            )
        }
        
        func contextMenuInteraction(
            _ interaction: UIContextMenuInteraction,
            willPerformPreviewActionForMenuWith configuration: UIContextMenuConfiguration,
            animator: UIContextMenuInteractionCommitAnimating
        ) {
            animator.addCompletion {
                self.view.didCommitView?()
            }
            
            guard !view.isPreviewOnly else { return }
            
            view.isActive = true
        }
        
        func contextMenuInteraction(
            _ interaction: UIContextMenuInteraction,
            willEndFor configuration: UIContextMenuConfiguration,
            animator: UIContextMenuInteractionAnimating?
        ) {
            print("willEnd")
        }
    }
}

struct PreviewContextAction {
    
    private let image: String?
    private let systemImage: String?
    private let attributes: UIMenuElement.Attributes
    private let action: (() -> ())?
    private let title: String
    
    init(
        title: String
    ) {
        self.init(title: title, image: nil, systemImage: nil, attributes: .disabled, action: nil)
    }

    init(
        title: String,
        attributes: UIMenuElement.Attributes = [],
        action: @escaping () -> ()
    ) {
        self.init(title: title, image: nil, systemImage: nil, attributes: attributes, action: action)
    }

    init(
        title: String,
        systemImage: String,
        attributes: UIMenuElement.Attributes = [],
        action: @escaping () -> ()
    ) {
        self.init(title: title, image: nil, systemImage: systemImage, attributes: attributes, action: action)
    }
    
    init(
        title: String,
        image: String,
        attributes: UIMenuElement.Attributes = [],
        action: @escaping () -> ()
    ) {
        self.init(title: title, image: image, systemImage: nil, attributes: attributes, action: action)
    }

    private init(
        title: String,
        image: String?,
        systemImage: String?,
        attributes: UIMenuElement.Attributes,
        action: (() -> ())?
    ) {
        self.title = title
        self.image = image
        self.systemImage = systemImage
        self.attributes = attributes
        self.action = action
    }
    
    private var uiImage: UIImage? {
        if let image = image {
            return UIImage(named: image)
        } else if let systemImage = systemImage {
            return UIImage(systemName: systemImage)
        } else {
            return nil
        }
    }

    fileprivate var uiAction: UIAction {
        UIAction(
            title: title,
            image: uiImage,
            attributes: attributes) { _ in
            action?()
        }
    }
}

@resultBuilder
struct ButtonBuilder {
    
    public static func buildBlock(_ buttons: PreviewContextAction...) -> [PreviewContextAction] {
        buttons
    }
}
