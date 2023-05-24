//
//  Extensions.swift
//  CirGo
//
//  Created by Choi Wai Lap on 6/12/2021.
//

import Foundation
import SwiftUI

extension View {
    
    func readSize(onChange: @escaping (CGSize) -> Void) -> some View {
        overlay(
            GeometryReader { (proxy: GeometryProxy) in
                Color.clear
                    .preference(key: CGSizePreferenceKey.self, value: proxy.size)
            }
        )
        .onPreferenceChange(CGSizePreferenceKey.self, perform: onChange)
    }
    
    func readGeometry(
        id: String,
        space: String,
        onChange: @escaping (String, [String : CGRect]) -> Void
    ) -> some View {
        overlay(
            GeometryReader { (proxy: GeometryProxy) in
                Color
                    .clear
                    .preference(
                        key: CGRectPreferenceKey.self,
                        value: [id : proxy.frame(in: .named(space))]
                    )
            }
        )
        .onPreferenceChange(CGRectPreferenceKey.self) { (cache: [String : CGRect]) in
            onChange(id, cache)
        }
    }
    
}
