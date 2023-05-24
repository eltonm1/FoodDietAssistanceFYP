//
//  NativeSearchBar.swift
//  CirGo
//
//  Created by 徐煒立 on 27/8/2021.
//

import SwiftUI

extension View {
    func nativeSearchBar(
        text: Binding<String>,
        placeHolder: String,
        backgroundColor: Color? = nil
    ) -> some View {
        self.modifier(
            NativeSearchBarModifier(
                text: text,
                placeHolder: placeHolder,
                backgroundColor: backgroundColor
            )
        )
    }
}

struct NativeSearchBarModifier: ViewModifier {
    
    @Binding var text: String
    let placeHolder: String
    let backgroundColor: Color?
    
    func body(content: Content) -> some View {
        content.overlay(
            NativeSearchBarControllerRepresentable(
                text: $text,
                placeHolder: placeHolder,
                backgroundColor: backgroundColor
            )
            .frame(width: 0, height: 0)
        )
    }
}

struct NativeSearchBarControllerRepresentable: UIViewControllerRepresentable {
    
    @Binding var text: String
    let placeHolder: String
    let backgroundColor: Color?
        
    func makeUIViewController(context: Context) -> NativeSearchBarController {
        let viewController = NativeSearchBarController { (vc: UIViewController) in
            let searchController: UISearchController? = vc.navigationItem.searchController
            searchController?.searchBar.searchTextField.typingAttributes = NSAttribute {
                FontKey.font(.regular, .body)
            }
            searchController?.searchBar.placeholder = placeHolder
            searchController?.searchBar.searchTextField.font = ProductSans.regular.uiFont(relativeTo: .body)
            if searchController != nil { return }
            vc.navigationItem.searchController = context.coordinator.searchController
            vc.view.backgroundColor = backgroundColor?.toUIColor()
        }
        context.coordinator.searchController.searchResultsUpdater = context.coordinator
        
        context.coordinator.viewController = viewController
        
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: NativeSearchBarController, context: Context) {
        
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UISearchResultsUpdating {
        
        var parent: NativeSearchBarControllerRepresentable
        var viewController: NativeSearchBarController!
        
        let searchController: UISearchController = UISearchController(searchResultsController: nil)
        
        init(_ parent: NativeSearchBarControllerRepresentable) {
            self.parent = parent
            searchController.obscuresBackgroundDuringPresentation = false
        }
        
        func updateSearchResults(for searchController: UISearchController) {
            if let searchBarText: String = searchController.searchBar.text {
                parent.text = searchBarText
            }
        }

    }
}

class NativeSearchBarController: UIViewController {
    let updateHandler: (UIViewController) -> Void
    
    init(onUpdate: @escaping (UIViewController) -> Void) {
        self.updateHandler = onUpdate
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError()
    }
    
    override func didMove(toParent parent: UIViewController?) {
        super.didMove(toParent: parent)
        if let parent: UIViewController = parent {
            updateHandler(parent)
        }
    }
}
