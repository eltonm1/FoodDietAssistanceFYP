//
//  MenuItem.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 13/1/2023.
//

import SwiftUI

protocol LabelItem {
    var label: String? { get }
    var systemName: String { get }
    var id: String { get }
}

struct MenuItem: LabelItem {
    var label: String?
    var systemName: String
    var children: [LabelItem]
    var id: String { "\(label ?? "")-\(systemName)" }

    init(label: String? = nil, systemName: String, @MenuBuilder children: () -> [LabelItem]) {
        self.label = label
        self.systemName = systemName
        self.children = children()
    }

//    init<T: LabelItem>(label: String? = nil, systemName: String, @MenuBuilder children: () -> [T]) {
//        self.label = label
//        self.systemName = systemName
//        self.children = children().map { $0 as any LabelItem }
//    }
//
//    init(label: String? = nil, systemName: String, @MenuBuilder child: () -> any LabelItem) {
//        self.label = label
//        self.systemName = systemName
//        self.children = [child()]
//    }
//
//    init(label: String? = nil, systemName: String, @MenuBuilder optionalChildren: () -> [any LabelItem]?) {
//        self.label = label
//        self.systemName = systemName
//        self.children = optionalChildren() ?? []
//    }
//
//    init(label: String? = nil, systemName: String, @MenuBuilder optionalChild: () -> (any LabelItem)?) {
//        self.label = label
//        self.systemName = systemName
//        self.children = [optionalChild()].compactMap { $0 }
//    }
}

struct ButtonItem: LabelItem {
    var label: String?
    var systemName: String
    var id: String { "\(label ?? "")-\(systemName)" }
    var action: () -> ()

    init(label: String? = nil, systemName: String, action: @escaping () -> ()) {
        self.label = label
        self.systemName = systemName
        self.action = action
    }
}

@resultBuilder
struct MenuBuilder {
    static func buildBlock(_ items: LabelItem...) -> [LabelItem] {
        items
    }
    
    static func buildBlock(_ components: [LabelItem]...) -> [LabelItem] {
        components.flatMap { $0 }
    }
}

// extension MenuBuilder {
//     static func buildIf(_ item: (any LabelItem)?) -> (any LabelItem)? {
//         item
//     }

//     static func buildEither(first item: any LabelItem) -> any LabelItem {
//         item
//     }
    
//     static func buildEither(second item: any LabelItem) -> any LabelItem {
//         item
//     }

//     static func buildOptional(_ item: (any LabelItem)?) -> (any LabelItem)? {
//         item
//     }

//     static func buildArray(_ items: [any LabelItem]) -> [any LabelItem] {
//         items
//     }

//     static func buildFinalResult(_ item: any LabelItem) -> any LabelItem {
//         item
//     }

//     static func buildExpression(_ item: any LabelItem) -> any LabelItem {
//         item
//     }

//     // Closure
//     static func buildExpression(_ item: () -> any LabelItem) -> any LabelItem {
//         item()
//     }

//     static func buildExpression(_ item: () -> [any LabelItem]) -> [any LabelItem] {
//         item()
//     }
// }
