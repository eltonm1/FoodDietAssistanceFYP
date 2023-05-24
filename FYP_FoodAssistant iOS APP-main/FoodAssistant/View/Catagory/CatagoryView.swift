//
//  ProductView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/11/2022.
//

import SwiftUI
import Introspect

struct CatagoryView: View {
    
    @EnvironmentObject var mvm: MainViewModel
    @StateObject var cvm: CatagoryViewModel
    @Namespace var ns
    let screenSize: CGSize
    let onScanBarcodeViewLoad: () -> Void
    let onScanBarcodeViewUnload: () -> Void
    
    init(
        foodDataService: FoodProductDataService,
        screenSize: CGSize,
        onScanBarcodeViewLoad: @escaping () -> Void,
        onScanBarcodeViewUnload: @escaping () -> Void
    ) {
        self._cvm = StateObject(wrappedValue: CatagoryViewModel(foodService: foodDataService))
        self.screenSize = screenSize
        self.onScanBarcodeViewLoad = onScanBarcodeViewLoad
        self.onScanBarcodeViewUnload = onScanBarcodeViewUnload
    }
    
    var body: some View {
        NavigationStack(path: $cvm.navigationPath) {
            ZStack {
                if cvm.foodsService.isLoading {
                    LoadingView()
                } else {
                    if cvm.viewType == .list {
                        ListView(
                            filteredCats: cvm.filteredCats,
                            products: cvm.catProductsDict,
                            colors: cvm.colors,
                            ns: ns,
                            screenHeight: screenSize.height,
                            onRefresh: cvm.onRefresh
                        )
                    } else {
                        GalleryView(
                            ns: ns,
                            filteredCats: cvm.filteredCats,
                            products: cvm.catProductsDict,
                            colors: cvm.colors,
                            screenHeight: screenSize.height
                        )
                    }
                }
            }
            .navigationTitle("Catagory")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    MenuView(menuItem: cvm.toolBarItem)
                }
            }
            .searchable(text: cvm.searchedCatagory, prompt: "e.g. Coke - Bottle 1.25L")
//            .nativeSearchBar(
//                text: cvm.searchedCatagory,
//                placeHolder: "Search Catagory",
//                backgroundColor: .systemGroupedBackground
//            )
            .searchSuggestions {
                SearchSuggestion(
                    path: $cvm.navigationPath,
                    search: cvm.searchedCatagory,
                    products: cvm.foodsService.products,
                    onClick: cvm.onSearchSuggestionClicked,
                    onEnter: { cvm.onNavigateToInputView(mvm: mvm, isEntering: true) },
                    onLeave: { cvm.onNavigateToInputView(mvm: mvm, isEntering: false) }
                )
            }
            .navigationDestination(for: CatagoryViewModel.NavigationRoute.self) {
                (detail: CatagoryViewModel.NavigationRoute) in
                switch detail {
                case .categoryDetailView(let categoryDetail):
                    CatagoryDetailView(detail: categoryDetail, screenHeight: screenSize.height)
                case .inputProductDetailView(let detail):
                    InputProductDetailView(detail: detail)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .top) {
                if let error: String = cvm.foodsService.errorMessage {
                    ErrorView(message: error, ns: ns)
                        .padding(.horizontal)
                }
            }
        }
        .productLargeNavigationBar()
        .frame(width: screenSize.width, height: screenSize.height)
    }
}

fileprivate struct ErrorView: View {
    let message: String
    let ns: Namespace.ID
    
    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .matchedGeometryEffect(id: "error-icon", in: ns)
                .foregroundColor(.white)
                .padding(4)
            Text(message)
                .foregroundColor(.white)
                .productFont(.regular, relativeTo: .body)
                .matchedGeometryEffect(id: "error-context", in: ns)
                .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .padding()
        .background(.systemRed)
        .cornerRadius(8, style: .continuous)
        .matchedGeometryEffect(id: "error-background", in: ns)
        .listRowBackground(EmptyView())
        .listRowInsets(.zero)
        .transition(.move(edge: .top))
    }
}

fileprivate struct ListView: View {
    let filteredCats: [String]
    let products: [String : [Product]]
    let colors: [String : Color]
    let ns: Namespace.ID
    let screenHeight: CGFloat
    let onRefresh: () -> Void
    
    var body: some View {
        List {
            Section {
                ForEach(filteredCats, id: \.self) { (cat: String) in
                    CatagoryListRow(
                        category: cat,
                        ns: ns,
                        products: products[cat] ?? [],
                        color: colors[cat] ?? .black,
                        screenHeight: screenHeight
                    )
                }
            } footer: {
                Rectangle()
                    .frame(height: screenHeight/8)
                    .opacity(0)
            }
        }
        .refreshable {
            onRefresh()
        }
        .edgesIgnoringSafeArea(.bottom)
    }
    
    fileprivate struct CatagoryListRow: View {
        let category: String
        let ns: Namespace.ID
        let products: [Product]
        let color: Color
        let screenHeight: CGFloat
        
        @ViewBuilder
        func destination(isPreview: Bool) -> some View {
            CatagoryDetailView(
                detail: CatagoryDetailView.CategoryDetail(
                    category: category,
                    products: products,
                    color: color,
                    isPreview: isPreview
                ),
                screenHeight: screenHeight
            )
        }
        
        var body: some View {
            HStack {
                Image(systemName: "circle.fill")
                    .foregroundColor(color)
                    .matchedGeometryEffect(id: "\(category)-icon", in: ns)
                Text("\(category)")
                    .productFont(.regular, relativeTo: .body)
                    .matchedGeometryEffect(id: "\(category)-catName", in: ns)
                Spacer()
                Text("\(products.count) items")
                    .foregroundColor(.secondary)
                    .productFont(.regular, relativeTo: .callout)
                    .matchedGeometryEffect(id: "\(category)-itemsNum", in: ns)
            }
            .matchedGeometryEffect(id: "\(category)-row", in: ns)
            .previewContextMenu(
                destination: destination(isPreview: false),
                preview: destination(isPreview: true),
                navigationValue: CatagoryViewModel
                    .NavigationRoute
                    .categoryDetailView(
                        CatagoryDetailView.CategoryDetail(
                            category: category,
                            products: products,
                            color: color,
                            isPreview: false
                        )
                    )
            )
        }
    }
}

fileprivate struct GalleryView: View {
    var ns: Namespace.ID
    let filteredCats: [String]
    let products: [String : [Product]]
    let colors: [String : Color]
    let screenHeight: CGFloat
    
    var body: some View {
        ScrollView {
            LazyVGrid(
                columns: [GridItem(.adaptive(100))],
                spacing: 16) {
                    ForEach(filteredCats, id: \.self) { (cat: String) in
                        CatagoryGalleryBlock(
                            category: cat,
                            ns: ns,
                            products: products[cat] ?? [],
                            color: colors[cat] ?? .black,
                            screenHeight: screenHeight
                        )
                    }
                }
                .padding(.bottom, screenHeight/8)
        }
        .edgesIgnoringSafeArea(.bottom)
    }
    
    fileprivate struct CatagoryGalleryBlock: View {
        let category: String
        let ns: Namespace.ID
        let products: [Product]
        let color: Color
        let screenHeight: CGFloat
        
        @ViewBuilder
        func destination(isPreview: Bool) -> some View {
            CatagoryDetailView(
                detail: CatagoryDetailView.CategoryDetail(
                    category: category,
                    products: products,
                    color: color,
                    isPreview: isPreview
                ),
                screenHeight: screenHeight
            )
        }
        
        var body: some View {
            VStack {
                Rectangle()
                    .foregroundColor(color)
                    .frame(height: 75)
                    .matchedGeometryEffect(id: "\(category)-icon", in: ns)
                Text("\(category)")
                    .foregroundColor(.primary)
                    .productFont(.regular, relativeTo: .body)
                    .matchedGeometryEffect(id: "\(category)-catName", in: ns)
                Text("\(products.count) items")
                    .foregroundColor(.secondary)
                    .productFont(.regular, relativeTo: .callout)
                    .matchedGeometryEffect(id: "\(category)-itemsNum", in: ns)
                Spacer()
            }
            .matchedGeometryEffect(id: "\(category)-row", in: ns)
            .previewContextMenu(
                destination: destination(isPreview: false),
                preview: destination(isPreview: true),
                navigationValue: CatagoryViewModel
                    .NavigationRoute
                    .categoryDetailView(
                        CatagoryDetailView.CategoryDetail(
                            category: category,
                            products: products,
                            color: color,
                            isPreview: false
                        )
                    )
            )
        }
    }
}

fileprivate struct LoadingView: View {
    var body: some View {
        VStack {
            ProgressView()
                .progressViewStyle(.circular)
        }
    }
}

struct ProductView_Previews: PreviewProvider {
    static var previews: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            CatagoryView(
                foodDataService: FoodProductDataService(),
                screenSize: size,
                onScanBarcodeViewLoad: { },
                onScanBarcodeViewUnload: { }
            )
        }
    }
}
