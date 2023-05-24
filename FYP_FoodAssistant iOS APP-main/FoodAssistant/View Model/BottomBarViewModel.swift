//
//  BottomBarViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 6/12/2022.
//

import Foundation
import UIKit
import Combine
import SwiftUI

class BottomBarViewModel: NSObject, ObservableObject {
    
    enum PageNumber: CGFloat {
        case one = 0, two = 1, three = 2
    }
    
    @Published var currentPageNumber: PageNumber = .two
    @Published private var tabOffset: CGFloat = 0
    @Published var normalizedCurrentTabOffset: CGFloat = 1 // [0, 1, 2]
    @Published var showBar = true
    @Published var pageChange = true // just a dummy publisher

    private var cancellables = Set<AnyCancellable>()
        
    var tabSV: UIScrollView?
    var viewWidth: CGFloat = .zero
    
    var tabScrollProgress: CGFloat { // [0, 1, 0]
        let progress: CGFloat = normalizedCurrentTabOffset <= 1 ? normalizedCurrentTabOffset : 2 - normalizedCurrentTabOffset
        return max(0, min(2, progress))
    }
    
    func scrollTo(page: PageNumber, animated: Bool) {
        let decodedOffset: CGFloat = page.rawValue * viewWidth
        tabOffset = decodedOffset
        normalizedCurrentTabOffset = tabOffset/viewWidth
        currentPageNumber = page
//        print("decodedOffset: \(page.rawValue)x\(viewWidth)=\(decodedOffset), normalizedCurrentTabOffset: \(tabOffset)/\(viewWidth)=\(normalizedCurrentTabOffset)")
        tabSV?.setContentOffset(
            .init(x: decodedOffset, y: 0),
            animated: animated
        )
    }   
    
    func onScrollViewDidEndScrolling() {
        if normalizedCurrentTabOffset < 0.5 {
            currentPageNumber = .one
            tabOffset = .zero
            normalizedCurrentTabOffset = 0
        } else if normalizedCurrentTabOffset < 1.5 {
            currentPageNumber = .two
            tabOffset = viewWidth
            normalizedCurrentTabOffset = 1
        } else {
            currentPageNumber = .three
            tabOffset = viewWidth*2
            normalizedCurrentTabOffset = 2
        }
    }
    
    func setSrollable(to scrollable: Bool) {
        self.tabSV?.isScrollEnabled = scrollable
    }
}

extension BottomBarViewModel: UIScrollViewDelegate {
    func scrollViewDidScroll(_ scrollView: UIScrollView) {
        guard let tabSV, scrollView.isDragging else { return }
        tabOffset = tabSV.contentOffset.x
//        print("tabOffset: \(tabOffset)")
        
//        let screenWidth: CGFloat = scrollView.frame.width
        normalizedCurrentTabOffset = tabOffset/viewWidth
//        print("normalizedCurrentTabOffset: \(tabOffset)/\(viewWidth)=\(normalizedCurrentTabOffset), isDragging: \(scrollView.isDragging)")
    }
    
    func scrollViewDidEndDecelerating(_ scrollView: UIScrollView) {
        onScrollViewDidEndScrolling()
        pageChange.toggle()
    }
    
    func scrollViewDidEndDragging(_ scrollView: UIScrollView, willDecelerate decelerate: Bool) {
//        if (!decelerate) {
            onScrollViewDidEndScrolling()
//        }
    }
}

