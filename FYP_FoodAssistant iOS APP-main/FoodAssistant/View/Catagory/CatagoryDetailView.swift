//
//  CatagoryDetailView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 3/1/2023.
//

import SwiftUI

struct CatagoryDetailView: View {
    
    struct CategoryDetail: Hashable {
        let category: String
        let products: [Product]
        let color: Color
        let isPreview: Bool
    }
    
    @EnvironmentObject var mvm: MainViewModel
    @StateObject var cdvm: CatagoryDetailViewModel
    @Namespace var ns
    
    let detail: CategoryDetail
    let screenHeight: CGFloat
    
    init(
        detail: CategoryDetail,
        screenHeight: CGFloat
    ) {
        self._cdvm = StateObject(wrappedValue: CatagoryDetailViewModel(products: detail.products))
        self.detail = detail
        self.screenHeight = screenHeight
    }
    
    var body: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            ScrollView(.vertical, showsIndicators: false) {
                AlphabetSessions(
                    characters: cdvm.characters,
                    ns: ns,
                    color: detail.color,
                    isExpanded: cdvm.expandedDict,
                    toggleExpanded: cdvm.toggleExpand(of:),
                    onRectUpdate: { (size: CGSize, rect: CGRect) in
                        guard !detail.isPreview, size != .zero else { return }
                        let size: CGSize = .init(width: size.width, height: size.height - screenHeight/8)
                        // MARK: Whenever Scrolling Does
                        // Resetting Timeout
                        if cdvm.hideIndicatorLabel && rect.minY < 0 {
                            cdvm.scrollerTimeOut = 0
                            cdvm.hideIndicatorLabel = false
                        }
                        
                        let rectHeight: CGFloat = rect.height
                        let viewHeight: CGFloat = size.height + cdvm.startOffset
                        
                        cdvm.scrollerHeight = (viewHeight/rectHeight)*viewHeight
                        
                        // MARK: Finding Scroll Indicator Offset
                        let progress = rect.minY / (rectHeight - size.height)
                        // MARK: Simply Multiply With View Height
                        // Eliminating Scroller Height
                        cdvm.indicatorOffset = -progress * (size.height - cdvm.scrollerHeight)
                    },
                    onRectsUpdate: { (pc: CatagoryDetailViewModel.ProductCharacter, minY: CGFloat) in
                        // MARK: Verifying Which section is at the Top (Near NavBar)
                        // Updating Character Rect When ever it's Updated
                        if cdvm.characters.indices.contains(pc.index) {
                            cdvm.characters[pc.index].minY = minY
                        }
                        
                        // Since Every Character moves up and goes beyond Zero (It will be like A,B,C,D)
                        // So We're taking the last character
                        if let last: CatagoryDetailViewModel.ProductCharacter =
                            cdvm.characters.last(where: {
                                (char: CatagoryDetailViewModel.ProductCharacter) in
                                char.minY < 0
                            }), last != cdvm.currentCharacter {
                            cdvm.currentCharacter = last
                        }
                    },
                    filteredProducts: cdvm.filteredProductsDict,
                    parentSize: size,
                    onEnterInputView: {
                        cdvm.onNavigateToInputView(mvm: mvm, isEntering: true)
                    },
                    onBackFromInputView: {
                        cdvm.onNavigateToInputView(mvm: mvm, isEntering: false)
                    }
                )
                .equatable()
            }
            .overlay(alignment: .topTrailing) {
                if !detail.isPreview {
                    ScrollerView(
                        title: cdvm.scrollerTitle,
                        subtitle: cdvm.scrollSubtitle,
                        color: detail.color,
                        shouldDisappear: cdvm.scrollerShouldDisappear,
                        height: cdvm.scrollerHeight,
                        yOffset: cdvm.indicatorOffset
                    )
                }
            }
            .padding(.bottom, screenHeight/8)
            .coordinateSpace(name: "scroller")
        }
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                MenuView(menuItem: cdvm.toolBarItem)
            }
        }
        .nativeSearchBar(
            text: $cdvm.searchedProduct,
            placeHolder: "Search Product"
        )
        .navigationTitle(detail.category)
        .productLargeNavigationBar()
        .readGeometry(
            id: "CategoryDetailView.GeometryReader",
            space: "scroller"
        ) { (id: String, cache: [String : CGRect]) in
            guard let rect: CGRect = cache[id] else { return }
            if cdvm.startOffset != rect.minY {
                cdvm.startOffset = rect.minY
            }
        }
        .onReceive(
            Timer
                .publish(
                    every: 0.01,
                    on: .main,
                    in: .default
                )
                .autoconnect()
        ) { _ in cdvm.onTimerUpdate() }
        .edgesIgnoringSafeArea(.bottom)
        .background(.adaptable(light: .white, dark: .black))
    }
}

fileprivate struct AlphabetSessions: View, Equatable {
    static func == (lhs: AlphabetSessions, rhs: AlphabetSessions) -> Bool {
        lhs.isExpanded == rhs.isExpanded &&
        lhs.filteredProducts == rhs.filteredProducts &&
        lhs.parentSize == rhs.parentSize
    }
    typealias PC = CatagoryDetailViewModel.ProductCharacter
    let characters: [PC]
    let ns: Namespace.ID
    let color: Color
    let isExpanded: [PC : Bool]
    let toggleExpanded: (PC) -> Void
    let onRectUpdate: (CGSize, CGRect) -> Void
    let onRectsUpdate: (PC, CGFloat) -> Void
    let filteredProducts: [PC : [Product]]
    let parentSize: CGSize
    let onEnterInputView: () -> Void
    let onBackFromInputView: () -> Void
    var body: some View {
        LazyVStack(spacing: 0) {
            ForEach(characters) { (pc: PC) in
                AlphabetSession(
                    isExpanded: isExpanded[pc] ?? false,
                    toggleExpanded: {
                        toggleExpanded(pc)
                    },
                    onRectUpdate: { (minY: CGFloat) in
                        onRectsUpdate(pc, minY)
                    },
                    products: filteredProducts[pc] ?? [],
                    color: color,
                    label: pc.value,
                    ns: ns,
                    onEnterInputView: onEnterInputView,
                    onBackFromInputView: onBackFromInputView,
                    oldMinY: pc.minY
                )
                .equatable()
            }
        }
        .background {
            GeometryReader {
                let rect: CGRect = $0.frame(in: .named("scroller"))
                Color.clear.task(id: rect) { onRectUpdate(parentSize, rect) }
            }
        }
    }
}

fileprivate struct AlphabetSession: View, Equatable {
    static func ==(lhs: AlphabetSession, rhs: AlphabetSession) -> Bool {
        lhs.isExpanded == rhs.isExpanded &&
        lhs.products == rhs.products &&
        lhs.oldMinY == rhs.oldMinY
    }
    let isExpanded: Bool
    let toggleExpanded: () -> ()
    let onRectUpdate: (CGFloat) -> ()
    let products: [Product]
    let color: Color
    let label: String
    let ns: Namespace.ID
    let onEnterInputView: () -> Void
    let onBackFromInputView: () -> Void
    let oldMinY: CGFloat
    var body: some View {
        VStack(alignment: .leading) {
            Button {
               toggleExpanded()
            } label: {
                Text(label)
                    .foregroundColor(.primary)
                    .productFont(.bold, relativeTo: .title2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .overlay(alignment: .trailing) {
                        Image(systemName: "chevron.up")
                            .resizable()
                            .scaledToFit()
                            .rotationEffect(.degrees(isExpanded ? 0 : 180))
                            .foregroundColor(.systemBlue)
                            .padding(8)
                    }
                    .padding(.horizontal, 24)
            }
            .padding(.top)
            if isExpanded {
                ForEach(products) { (p: Product) in
                    ProductInformationRow(
                        ns: ns,
                        product: p,
                        color: color,
                        onEnterInputView: onEnterInputView,
                        onBackFromInputView: onBackFromInputView
                    )
                }
            }
            TinyDivider()
        }
        .background {
            GeometryReader { (proxy: GeometryProxy) in
                let minY: CGFloat = proxy.frame(in: .named("scroller")).minY
                Color.clear.task(id: minY.isLess(than: .zero)) { onRectUpdate(minY) }
            }
        }
    }
}

fileprivate struct TinyDivider: View {
    var body: some View {
        Rectangle()
            .frame(height: 0.5)
            .foregroundColor(.secondary)
    }
}

struct CatagoryDetailView_Previews: PreviewProvider {
    @StateObject static var mvm = MainViewModel()
    @StateObject static var cvm = CatagoryViewModel(foodService: FoodProductDataService())
    static let category: String = "Beer / Wines / Spirits"
    static let detail = CatagoryDetailView.CategoryDetail(
        category: category,
        products: cvm.foodsService.productWhoweCategory(number: 1, is: category),
        color: .random,
        isPreview: false
    )
    static var previews: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let height: CGFloat = proxy.size.height
            NavigationStack {
                CatagoryDetailView(detail: detail, screenHeight: height)
            }
        }
    }
}
