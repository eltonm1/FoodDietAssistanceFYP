//
//  PreviewContextMenuExtension.swift
//  CirGo
//
//  Created by Choi Wai Lap on 13/7/2022.
//

import Foundation
import SwiftUI

extension View {
    func previewContextMenu<Preview: View, Destination: View, NavigationValue: Hashable>(
            destination: Destination,
            preview: Preview,
            navigationValue: NavigationValue? = nil,
            preferredContentSize: CGSize? = nil,
            presentAsSheet: Bool = false,
            didTapView: (() -> Void)? = nil,
            didCommitView: (() -> Void)? = nil,
            @ButtonBuilder actions: () -> [PreviewContextAction] = { [] }
        ) -> some View {
            modifier(
                PreviewContextViewModifier(
                    destination: destination,
                    preview: preview,
                    navigationValue: navigationValue,
                    preferredContentSize: preferredContentSize,
                    presentAsSheet: presentAsSheet,
                    didCommitView: didCommitView,
                    actions: actions
                )
            )
        }
        
    func previewContextMenu<Preview: View, NavigationValue: Hashable>(
        preview: Preview,
        navigationValue: NavigationValue? = nil,
        preferredContentSize: CGSize? = nil,
        presentAsSheet: Bool = false,
        didTapView: (() -> Void)? = nil,
        didCommitView: (() -> Void)? = nil,
        @ButtonBuilder actions: () -> [PreviewContextAction] = { [] }
    ) -> some View {
        modifier(
            PreviewContextViewModifier<Preview, EmptyView, NavigationValue>(
                preview: preview,
                navigationValue: navigationValue,
                preferredContentSize: preferredContentSize,
                presentAsSheet: presentAsSheet,
                didCommitView: didCommitView,
                actions: actions
            )
        )
    }
    
    func previewContextMenu<Destination: View>(
        destination: Destination,
        preferredContentSize: CGSize? = nil,
        presentAsSheet: Bool = false,
        didTapView: (() -> Void)? = nil,
        didCommitView: (() -> Void)? = nil,
        @ButtonBuilder actions: () -> [PreviewContextAction] = { [] }
    ) -> some View {
        modifier(
            PreviewContextViewModifier<EmptyView, Destination, Int>(
                destination: destination,
                preferredContentSize: preferredContentSize,
                presentAsSheet: presentAsSheet,
                didCommitView: didCommitView,
                actions: actions
            )
        )
    }

    @ViewBuilder
    func `if`<Content: View>(
        _ conditional: Bool,
        @ViewBuilder content: (Self) -> Content
    ) -> some View {
        if conditional {
            content(self)
        } else {
            self
        }
    }
    
    @ViewBuilder
    func `if`<TrueContent: View, FalseContent: View>(
        _ conditional: Bool,
        @ViewBuilder if ifContent: (Self) -> TrueContent,
        @ViewBuilder else elseContent: (Self) -> FalseContent
    ) -> some View {
        if conditional {
            ifContent(self)
        } else {
            elseContent(self)
        }
    }
    
    @ViewBuilder
    func ifLet<Value, Content: View>(
        _ value: Value?,
        @ViewBuilder content: (Self, Value) -> Content
    ) -> some View {
        if let value = value {
            content(self, value)
        } else {
            self
        }
    }
}
