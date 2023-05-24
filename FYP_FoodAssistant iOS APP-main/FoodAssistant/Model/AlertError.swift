//
//  AlertError.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 9/11/2022.
//

import Foundation

struct AlertError {
    
    var title: String = ""
    var message: String = ""
    var primaryButtonTitle: String = "Accept"
    var secondaryButtonTitle: String?
    var primaryAction: (() -> ())? = nil
    var secondaryAction: (() -> ())? = nil
    
    init(
        title: String = "",
        message: String = "",
        primaryButtonTitle: String = "Accept",
        secondaryButtonTitle: String? = nil,
        primaryAction: (() -> ())? = nil,
        secondaryAction: (() -> ())? = nil
    ) {
        self.title = title
        self.message = message
        self.primaryAction = primaryAction
        self.primaryButtonTitle = primaryButtonTitle
        self.secondaryAction = secondaryAction
    }
    
}
