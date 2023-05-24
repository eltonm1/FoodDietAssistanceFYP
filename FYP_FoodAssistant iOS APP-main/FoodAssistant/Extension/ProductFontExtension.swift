//
//  ProductFontExtension.swift
//  CirGo
//
//  Created by Choi Wai Lap on 13/7/2022.
//

import SwiftUI
import SwiftUIX
import Introspect
import Foundation

enum ProductSans: String {
    case boldItalic = "Product Sans Bold Italic"
    case bold = "Product Sans Bold"
    case regular = "Product Sans"
    case italic = "Product Sans Italic"
    
    func uiFont(size: CGFloat) -> UIFont {
        UIFont(name: self.rawValue, size: size)!
    }
    
    func uiFont(relativeTo: Font.TextStyle) -> UIFont {
        UIFont(name: self.rawValue, size: relativeTo.defaultMetrics.size)!
    }
}

extension View {
    func productFont(_ font: ProductSans = .bold, category: ContentSizeCategory) -> some View {
        self.font(.custom(font.rawValue, size: category.size))
    }
    
    func productFont(_ font: ProductSans = .bold, size: CGFloat) -> some View {
        self.font(.custom(font.rawValue, size: size))
    }
    
    func productFont(_ font: ProductSans = .bold, relativeTo: Font.TextStyle = .title) -> some View {
        self.font(.custom(font.rawValue, relativeTo: relativeTo))
    }
    
    func productLargeNavigationBar() -> some View {
        self.introspectNavigationController { (vc: UINavigationController) in
            vc.navigationBar.titleTextAttributes = NSAttribute {
                FontKey.font(.bold, .headline)
            }
            vc.navigationBar.largeTitleTextAttributes = NSAttribute {
                FontKey.font(.bold, .largeTitle)
            }
        }
    }
}

extension Text {
    func productFont(_ font: ProductSans, relativeTo: Font.TextStyle = .title) -> Text {
        self.font(.custom(font.rawValue, relativeTo: relativeTo))
    }
}

extension String {
    func productAttribute(_ font: ProductSans, size: CGFloat) -> NSAttributedString {
        NSAttributedString(string: self, attributes: [NSAttributedString.Key.font : font.uiFont(size: size)])
    }
    
    func productAttribute(_ font: ProductSans, size: CGFloat, color: Color) -> NSAttributedString {
        NSAttributedString(string: self, attributes: [NSAttributedString.Key.font : font.uiFont(size: size), NSAttributedString.Key.foregroundColor : UIColor(color)])
    }
    
    func productAttribute(_ font: ProductSans, relativeTo: Font.TextStyle) -> NSAttributedString {
        NSAttributedString(string: self, attributes: [.font : font.uiFont(size: relativeTo.defaultMetrics.size)])
    }
    
    func productAttribute(_ font: ProductSans, relativeTo: Font.TextStyle, color: Color) -> NSAttributedString {
        NSAttributedString(string: self, attributes: [.font : font.uiFont(size: relativeTo.defaultMetrics.size), .foregroundColor : color.toUIColor() ?? .black])
    }
    
    func productAttributeSize(_ font: ProductSans, relativeTo: Font.TextStyle) -> CGSize {
        productAttribute(font, relativeTo: relativeTo).size()
    }
    
    func productAttributeWidth(_ font: ProductSans, relativeTo: Font.TextStyle) -> CGFloat {
        productAttributeSize(font, relativeTo: relativeTo).width
    }
    
    func productAttributeHeight(_ font: ProductSans, relativeTo: Font.TextStyle) -> CGFloat {
        productAttributeSize(font, relativeTo: relativeTo).height
    }
}

@resultBuilder
struct AttributeBuilder {
    static func buildBlock(_ components: FontKey...) -> [NSAttributedString.Key : Any] {
        Dictionary(uniqueKeysWithValues: components.map { (fontKey: FontKey) in
            fontKey.nsAttributedStringKeyPair
        })
    }
}

func NSAttribute(@AttributeBuilder _ content: () -> [NSAttributedString.Key : Any]) -> [NSAttributedString.Key : Any] {
    content()
}

enum FontKey {
    case font(ProductSans, Font.TextStyle)
    case color(Color)
    
    var nsAttributedStringKeyPair: (NSAttributedString.Key, Any) {
        switch self {
        case .font(let font, let style):
            return (.font, font.uiFont(relativeTo: style))
        case .color(let color):
            return (.foregroundColor, UIColor(color))
        }
    }
}

extension ContentSizeCategory {
    var size: CGFloat {
        switch self {
        case .small:
            return 14
        case .medium:
            return 16
        case .large:
            return 20
        default:
            return 14
        }
    }
}

