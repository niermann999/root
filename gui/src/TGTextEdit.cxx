// @(#)root/gui:$Name:  $:$Id: TGTextEdit.cxx,v 1.3 2000/07/07 00:34:17 rdm Exp $
// Author: Fons Rademakers   3/7/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextEdit                                                           //
//                                                                      //
// A TGTextEdit is a specialization of TGTextView. It provides the      //
// text edit functionality to the static text viewing widget.           //
// For the messages supported by this widget see the TGView class.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGTextEdit.h"
#include "TSystem.h"
#include "TMath.h"
#include "TTimer.h"
#include "KeySymbols.h"


ClassImp(TGTextEdit)


//______________________________________________________________________________
TGTextEdit::TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, Int_t id,
                       UInt_t sboptions, ULong_t back) :
     TGTextView(parent, w, h, id, sboptions, back)
{
   // Create a text edit widget.

   Init();
}

//______________________________________________________________________________
TGTextEdit::TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, TGText *text,
                       Int_t id, UInt_t sboptions, ULong_t back) :
     TGTextView(parent, w, h, text, id, sboptions, back)
{
   // Create a text edit widget. Initialize it with the specified text buffer.

   Init();
}

//______________________________________________________________________________
TGTextEdit::TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h,
                       const char *string, Int_t id, UInt_t sboptions,
                       ULong_t back) :
     TGTextView(parent, w, h, string, id, sboptions, back)
{
   // Create a text edit widget. Initialize it with the specified string.

   Init();
}

//______________________________________________________________________________
TGTextEdit::~TGTextEdit()
{
   // Cleanup text edit widget.

   gVirtualX->DeleteGC(fCursor0GC);
   gVirtualX->DeleteGC(fCursor1GC);

   delete fCurBlink;
}

//______________________________________________________________________________
void TGTextEdit::Init()
{
   // Initiliaze a text edit widget.

   GCValues_t gval;
   fCursor1GC = gVirtualX->CreateGC(fCanvas->GetId(), 0);
   gVirtualX->CopyGC(fNormGC, fCursor1GC, 0);

   gval.fMask = kGCFunction;
   gval.fFunction = kGXand;
   gVirtualX->ChangeGC(fCursor1GC, &gval);

   fCursor0GC = gVirtualX->CreateGC(fCanvas->GetId(), 0);
   gVirtualX->CopyGC(fSelGC, fCursor0GC, 0);
   gval.fFunction = kGXxor;
   gVirtualX->ChangeGC(fCursor0GC, &gval);

   gVirtualX->SetCursor(fCanvas->GetId(), fgDefaultCursor);

   fCursorState = 1;
   fCurrent.fY  = fCurrent.fX = 0;
   fPasteOK     = kFALSE;
   fInsertMode  = kInsert;
   fCurBlink    = 0;
}

//______________________________________________________________________________
Long_t TGTextEdit::ReturnLongestLineWidth()
{
   // Return width of longest line in widget.

   Long_t linewidth = TGTextView::ReturnLongestLineWidth();
   linewidth += 3*fScrollVal.fX;
   return linewidth;
}

//______________________________________________________________________________
void TGTextEdit::Clear(Option_t *)
{
   // Clear text edit widget.

   fCursorState = 1;
   fCurrent.fY = fCurrent.fX = 0;
   TGTextView::Clear();
}

//______________________________________________________________________________
Bool_t TGTextEdit::SaveFile(const char *filename)
{
   // Save file.

   return fText->Save(filename);
}

//______________________________________________________________________________
Bool_t TGTextEdit::Cut()
{
   // Cut text.

   if (!Copy())
      return kFALSE;
   Delete();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::Paste()
{
   // Paste text into widget.

    gVirtualX->ConvertPrimarySelection(fId, fClipboard, 0);
    return kTRUE;
}

//______________________________________________________________________________
void TGTextEdit::Delete(Option_t *)
{
   // Delete selection.

   TGLongPosition pos;
   pos.fY = ToObjYCoord(fVisible.fY);
   if (!fIsMarked)
      return;
   fText->DelText(fMarkedStart, fMarkedEnd);
   UnMark();
   if (fMarkedStart.fY < pos.fY)
      pos.fY = fMarkedStart.fY;
   pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   if (fMarkedStart.fX < pos.fX)
      pos.fX = fMarkedStart.fX;
   SetVsbPosition((ToScrYCoord(pos.fY)+fVisible.fY)/fScrollVal.fY);
   SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX)/fScrollVal.fX);
   SetSBRange(kHorizontal);
   SetSBRange(kVertical);
   SetCurrent(fMarkedStart);

   SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED), fWidgetId, kFALSE);
}

//______________________________________________________________________________
Bool_t TGTextEdit::Search(const char *string, Bool_t direction,
                          Bool_t caseSensitive)
{
   // Search for string in the specified direction.

   TGLongPosition pos;
   if (!fText->Search(&pos, fCurrent, string, direction, caseSensitive))
      return kFALSE;
   UnMark();
   fIsMarked = kTRUE;
   fMarkedStart.fY = fMarkedEnd.fY = pos.fY;
   fMarkedStart.fX = pos.fX;
   fMarkedEnd.fX = fMarkedStart.fX + strlen(string)-1;
   fMarkedEnd.fX++;
   if (direction)
      SetCurrent(fMarkedEnd);
   else
      SetCurrent(fMarkedStart);
   fMarkedEnd.fX--;
   pos.fY = ToObjYCoord(fVisible.fY);
   if (fCurrent.fY < pos.fY ||
       ToScrYCoord(fCurrent.fY) >= (Int_t)fCanvas->GetHeight())
      pos.fY = fMarkedStart.fY;
   pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   if (fCurrent.fX < pos.fX ||
       ToScrXCoord(fCurrent.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())
      pos.fX = fMarkedStart.fX;

   SetVsbPosition((ToScrYCoord(pos.fY)+fVisible.fY)/fScrollVal.fY);
   SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX)/fScrollVal.fX);
   DrawRegion(0, (Int_t)ToScrYCoord(fMarkedStart.fY), fCanvas->GetWidth(),
              UInt_t(ToScrYCoord(fMarkedEnd.fY+1)-ToScrYCoord(fMarkedEnd.fY)));

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::Replace(TGLongPosition textPos, const char *oldText,
                           const char *newText, Bool_t direction, Bool_t caseSensitive)
{
   // Replace text starting at textPos.

   TGLongPosition pos;
   if (!fText->Replace(textPos, oldText, newText, direction, caseSensitive))
      return kFALSE;
   UnMark();
   fIsMarked = kTRUE;
   fMarkedStart.fY = fMarkedEnd.fY = textPos.fY;
   fMarkedStart.fX = textPos.fX;
   fMarkedEnd.fX = fMarkedStart.fX + strlen(newText)-1;
   fMarkedEnd.fX++;

   if (direction)
      SetCurrent(fMarkedEnd);
   else
      SetCurrent(fMarkedStart);
   fMarkedEnd.fX--;
   pos.fY = ToObjYCoord(fVisible.fY);
   if (fCurrent.fY < pos.fY ||
       ToScrYCoord(fCurrent.fY) >= (Int_t)fCanvas->GetHeight())
      pos.fY = fMarkedStart.fY;
   pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   if (fCurrent.fX < pos.fX ||
       ToScrXCoord(fCurrent.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())
      pos.fX = fMarkedStart.fX;

   SetVsbPosition((ToScrYCoord(pos.fY)+fVisible.fY)/fScrollVal.fY);
   SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX)/fScrollVal.fX);
   DrawRegion(0, (Int_t)ToScrYCoord(fMarkedStart.fY), fCanvas->GetWidth(),
              UInt_t(ToScrYCoord(fMarkedEnd.fY+1)-ToScrYCoord(fMarkedEnd.fY)));

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::Goto(Long_t line)
{
   // Goto the specified line.

   TGLongPosition pos;
   if (line-1 >= fText->RowCount())
      return kFALSE;
   pos.fY = line;
   pos.fX = 0;
   SetVsbPosition((ToScrYCoord(line-1)+fVisible.fY)/fScrollVal.fY);
   SetHsbPosition(0);
   SetCurrent(pos);

   return kTRUE;
}

//______________________________________________________________________________
void TGTextEdit::SetInsertMode(EInsertMode mode)
{
   // Sets the mode how characters are entered.

   if (fInsertMode == mode) return;

   fInsertMode = mode;
}

//______________________________________________________________________________
void TGTextEdit::CursorOff()
{
   // If cursor if on, turn it off.

   if (fCursorState == 1)
      DrawCursor(2);
   fCursorState = 2;
}

//______________________________________________________________________________
void TGTextEdit::CursorOn()
{
   // Turn cursor on.

   DrawCursor(1);
   fCursorState = 1;

   if (fCurBlink)
      fCurBlink->Reset();
}

//______________________________________________________________________________
void TGTextEdit::SetCurrent(TGLongPosition new_coord)
{
   // Make the specified position the current position.

   CursorOff();

   fCurrent.fY = new_coord.fY;
   fCurrent.fX = new_coord.fX;

   CursorOn();

   SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_DATACHANGE), fWidgetId, 0);
}

//______________________________________________________________________________
void TGTextEdit::DrawCursor(Int_t mode)
{
   // Draw cursor. If mode = 1 draw cursor, if mode = 2 erase cursor.

   char count = -1;
   char cursor = ' ';
   if (fCurrent.fY >= fText->RowCount() || fCurrent.fX > fText->GetLineLength(fCurrent.fY))
      return;

   if (fCurrent.fY >= ToObjYCoord(fVisible.fY) &&
       fCurrent.fY <= ToObjYCoord(fVisible.fY+fCanvas->GetHeight()) &&
       fCurrent.fX >= ToObjXCoord(fVisible.fX, fCurrent.fY) &&
       fCurrent.fX <= ToObjXCoord(fVisible.fX+fCanvas->GetWidth(),fCurrent.fY)) {
      if (fCurrent.fY < fText->RowCount())
         count = fText->GetChar(fCurrent);
      if (count == -1)
         cursor =' ';
      else
         cursor = count;
      if (mode == 2) {
         if (fIsMarked && count != -1) {
            if ((fCurrent.fY > fMarkedStart.fY && fCurrent.fY < fMarkedEnd.fY) ||
                (fCurrent.fY == fMarkedStart.fY && fCurrent.fX >= fMarkedStart.fX &&
                 fCurrent.fY < fMarkedEnd.fY) ||
                (fCurrent.fY == fMarkedEnd.fY && fCurrent.fX <= fMarkedEnd.fX &&
                 fCurrent.fY > fMarkedStart.fY) ||
                (fCurrent.fY == fMarkedStart.fY && fCurrent.fY == fMarkedEnd.fY &&
                 fCurrent.fX >= fMarkedStart.fX && fCurrent.fX <= fMarkedEnd.fX &&
                 fMarkedStart.fX != fMarkedEnd.fX)) {
               // back ground fillrectangle
               gVirtualX->FillRectangle(fCanvas->GetId(), fSelbackGC,
                                     Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                     Int_t(ToScrYCoord(fCurrent.fY)),
                                     UInt_t(ToScrXCoord(fCurrent.fX+1, fCurrent.fY) -
                                     ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                     UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
               if (count != -1)
                  gVirtualX->DrawString(fCanvas->GetId(), fSelGC, (Int_t)ToScrXCoord(fCurrent.fX,fCurrent.fY),
                       Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent), &cursor, 1);
            } else {
               gVirtualX->ClearArea(fCanvas->GetId(),
                                    Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                    Int_t(ToScrYCoord(fCurrent.fY)),
                                    UInt_t(ToScrXCoord(fCurrent.fX+1, fCurrent.fY) -
                                    ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                    UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
               if (count != -1)
                  gVirtualX->DrawString(fCanvas->GetId(), fNormGC, (Int_t)ToScrXCoord(fCurrent.fX,fCurrent.fY),
                       Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent), &cursor, 1);
            }
         } else {
//            gVirtualX->DrawLine(fCanvas->GetId(), fCursor0GC,
//                                ToScrXCoord(fCurrent.fX, fCurrent.fY),
//                                ToScrYCoord(fCurrent.fY),
//                                ToScrXCoord(fCurrent.fX, fCurrent.fY),
//                                ToScrYCoord(fCurrent.fY+1)-1);
            gVirtualX->FillRectangle(fCanvas->GetId(), fCursor0GC,
                                     Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                     Int_t(ToScrYCoord(fCurrent.fY)),
                                     2,
                                     UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
            gVirtualX->DrawString(fCanvas->GetId(), fNormGC, (Int_t)ToScrXCoord(fCurrent.fX,fCurrent.fY),
                       Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent), &cursor, 1);
         }
      } else
         if (mode == 1) {
//            gVirtualX->DrawLine(fCanvas->GetId(), fCursor1GC,
//                                ToScrXCoord(fCurrent.fX, fCurrent.fY),
//                                ToScrYCoord(fCurrent.fY),
//                                ToScrXCoord(fCurrent.fX, fCurrent.fY),
//                                ToScrYCoord(fCurrent.fY+1)-1);
            gVirtualX->FillRectangle(fCanvas->GetId(), fCursor1GC,
                                     Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                     Int_t(ToScrYCoord(fCurrent.fY)),
                                     2,
                                     UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
         }
   }
}

//______________________________________________________________________________
void TGTextEdit::AdjustPos()
{
   // Adjust current position.

   TGLongPosition pos;
   pos.fY = fCurrent.fY;
   pos.fX = fCurrent.fX;

   if (pos.fY < ToObjYCoord(fVisible.fY))
      pos.fY = ToObjYCoord(fVisible.fY);
   else if (ToScrYCoord(pos.fY+1) >= (Int_t) fCanvas->GetHeight())
      pos.fY = ToObjYCoord(fVisible.fY + fCanvas->GetHeight())-1;
   if (pos.fX < ToObjXCoord(fVisible.fX, pos.fY))
      pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   else if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t) fCanvas->GetWidth())
      pos.fX = ToObjXCoord(fVisible.fX + fCanvas->GetWidth(), pos.fY)-1;
   if (pos.fY != fCurrent.fY || pos.fX != fCurrent.fX)
      SetCurrent(pos);
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleTimer(TTimer *t)
{
   // Handle timer cursor blink timer.

   if (t != fCurBlink) {
      TGTextView::HandleTimer(t);
      return kTRUE;
   }

   if (fCursorState == 1)
      fCursorState = 2;
   else
      fCursorState = 1;

   DrawCursor(fCursorState);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleSelection(Event_t *event)
{
   // Handle selection notify event.

   TString data;
   Int_t   nchar;

   gVirtualX->GetPasteBuffer((Window_t)event->fUser[0], (Atom_t)event->fUser[3],
                             data, nchar, kFALSE);

   if (!nchar) return kTRUE;

   delete fClipText;

   fClipText = new TGText;
   fClipText->LoadBuffer(data.Data());

   TGLongPosition start_src, end_src, pos;

   pos.fX = pos.fY = 0;
   start_src.fY = start_src.fX = 0;
   end_src.fY = fClipText->RowCount()-1;
   end_src.fX = fClipText->GetLineLength(end_src.fY)-1;

   if (end_src.fX < 0)
      end_src.fX = 0;
   fText->InsText(fCurrent, fClipText, start_src, end_src);
   UnMark();
   pos.fY = fCurrent.fY + fClipText->RowCount()-1;
   pos.fX = fClipText->GetLineLength(fClipText->RowCount()-1);
   if (start_src.fY == end_src.fY)
      pos.fX = pos.fX + fCurrent.fX;
   SetCurrent(pos);
   if (ToScrYCoord(pos.fY) >= (Int_t)fCanvas->GetHeight())
      pos.fY = ToScrYCoord(pos.fY) + fVisible.fY - fCanvas->GetHeight()/2;
   else
      pos.fY = fVisible.fY;
   if (ToScrXCoord(pos.fX, fCurrent.fY) >= (Int_t) fCanvas->GetWidth())
      pos.fX = ToScrXCoord(pos.fX, fCurrent.fY) + fVisible.fX + fCanvas->GetWidth()/2;
   else if (ToScrXCoord(pos.fX, fCurrent.fY < 0) && pos.fX != 0) {
      if (fVisible.fX - (Int_t)fCanvas->GetWidth()/2 > 0)
         pos.fX = fVisible.fX - fCanvas->GetWidth()/2;
      else
         pos.fX = 0;
   } else
      pos.fX = fVisible.fX;

   SetSBRange(kHorizontal);
   SetSBRange(kVertical);
   SetVsbPosition(pos.fY/fScrollVal.fY);
   SetHsbPosition(pos.fX/fScrollVal.fX);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleButton(Event_t *event)
{
   // Handle mouse button event in text edit widget.

   if (event->fWindow != fCanvas->GetId())
      return kTRUE;

   TGLongPosition pos;

   TGTextView::HandleButton(event);

   if (event->fType == kButtonPress) {
      if (event->fCode == kButton1 || event->fCode == kButton2) {
         pos.fY = ToObjYCoord(fVisible.fY + event->fY);
         if (pos.fY >= fText->RowCount())
            pos.fY = fText->RowCount()-1;
         pos.fX = ToObjXCoord(fVisible.fX+event->fX, pos.fY);
         if (pos.fX >= fText->GetLineLength(pos.fY))
            pos.fX = fText->GetLineLength(pos.fY);
         while (fText->GetChar(pos) == 16)
            pos.fX++;

         SetCurrent(pos);
      }
      if (event->fCode == kButton2) {
         if (gVirtualX->GetPrimarySelectionOwner() != kNone)
            gVirtualX->ConvertPrimarySelection(fId, fClipboard, event->fTime);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in text edit widget.

   TGLongPosition pos;
   if (event->fWindow != fCanvas->GetId())
      return kTRUE;

   // TGTextView::HandleMotion(event);
   if (fScrolling == -1) {
      pos.fY = ToObjYCoord(fVisible.fY+event->fY);
      if (pos.fY >= fText->RowCount())
         pos.fY = fText->RowCount()-1;
      pos.fX = ToObjXCoord(fVisible.fX+event->fX, pos.fY);
      if (pos.fX > fText->GetLineLength(pos.fY))
         pos.fX = fText->GetLineLength(pos.fY);
      if (fText->GetChar(pos) == 16) {
         if (pos.fX < fCurrent.fX)
            pos.fX = fCurrent.fX;
         if (pos.fX > fCurrent.fX)
            do
               pos.fX++;
            while (fText->GetChar(pos) == 16);
      }
      event->fY = (Int_t)ToScrYCoord(pos.fY);
      event->fX = (Int_t)ToScrXCoord(pos.fX, pos.fY);
      if (pos.fY != fCurrent.fY || pos.fX != fCurrent.fX) {
         TGTextView::HandleMotion(event);
         SetCurrent(pos);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleKey(Event_t *event)
{
   // The key press event handler converts a key press to some line editor
   // action.

   Bool_t mark_ok = kFALSE;
   char   input[10];
   Int_t  n;
   UInt_t keysym;

   if (event->fType == kGKeyPress) {
      gVirtualX->LookupString(event, input, sizeof(input), keysym);
      n = strlen(input);

      AdjustPos();

      switch ((EKeySym)keysym) {   // ignore these keys
         case kKey_Shift:
         case kKey_Control:
         case kKey_Meta:
         case kKey_Alt:
         case kKey_CapsLock:
         case kKey_NumLock:
         case kKey_ScrollLock:
            return kTRUE;
         default:
            break;
      }
      if (event->fState & kKeyControlMask) {   // Cntrl key modifier pressed
         switch((EKeySym)keysym & ~0x20) {   // treat upper and lower the same
            case kKey_A:
               mark_ok = kTRUE;
               Home();
               break;
            case kKey_B:
               mark_ok = kTRUE;
               PrevChar();
               break;
            case kKey_C:
               Copy();
               if (fCurrent.fX == 0 && fCurrent.fY == fMarkedEnd.fY+1) {
                  TGLongPosition pos;
                  pos.fY = fClipText->RowCount();
                  pos.fX = 0;
                  fClipText->InsText(pos, 0);
               }
               return kTRUE;
            case kKey_D:
               if (fIsMarked)
                  Cut();
               else {
                  Long_t len = fText->GetLineLength(fCurrent.fY);
                  if (fCurrent.fY == fText->RowCount()-1 && fCurrent.fX == len) {
                     gVirtualX->Bell(0);
                     return kTRUE;
                  }
                  NextChar();
                  DelChar();
               }
               break;
            case kKey_E:
               mark_ok = kTRUE;
               End();
               break;
            case kKey_F:
               mark_ok = kTRUE;
               NextChar();
               break;
            case kKey_H:
               DelChar();
               break;
            case kKey_K:
               End();
               fIsMarked = kTRUE;
               Mark(fCurrent.fX, fCurrent.fY);
               Cut();
               break;
            case kKey_U:
               Home();
               UnMark();
               fMarkedStart.fY = fMarkedEnd.fY = fCurrent.fY;
               fMarkedStart.fX = fMarkedEnd.fX = fCurrent.fX;
               End();
               fIsMarked = kTRUE;
               Mark(fCurrent.fX, fCurrent.fY);
               Cut();
               break;
            case kKey_V:
            case kKey_Y:
               Paste();
               return kTRUE;
            case kKey_X:
               Cut();
               return kTRUE;
            default:
               return kTRUE;
         }
      }
      if (n && keysym >= 32 && keysym < 127 &&     // printable keys
          !(event->fState & kKeyControlMask) &&
          (EKeySym)keysym != kKey_Delete &&
          (EKeySym)keysym != kKey_Backspace) {

         if (fIsMarked)
            Cut();
         InsChar(input[0]);

      } else {

         switch ((EKeySym)keysym) {
            case kKey_F3:
               SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_F3), fWidgetId,
                           kTRUE);
               break;
            case kKey_Delete:
               if (fIsMarked)
                  Cut();
               else {
                  Long_t len = fText->GetLineLength(fCurrent.fY);
                  if (fCurrent.fY == fText->RowCount()-1 && fCurrent.fX == len) {
                     gVirtualX->Bell(0);
                     return kTRUE;
                  }
                  NextChar();
                  DelChar();
               }
               break;
            case kKey_Return:
            case kKey_Enter:
               BreakLine();
               break;
            case kKey_Tab:
               InsChar('\t');
               break;
            case kKey_Backspace:
               if (fIsMarked)
                  Cut();
               else
                  DelChar();
               break;
            case kKey_Left:
               mark_ok = kTRUE;
               PrevChar();
               break;
            case kKey_Right:
               mark_ok = kTRUE;
               NextChar();
               break;
            case kKey_Up:
               mark_ok = kTRUE;
               LineUp();
               break;
            case kKey_Down:
               mark_ok = kTRUE;
               LineDown();
               break;
            case kKey_PageUp:
               mark_ok = kTRUE;
               ScreenUp();
               break;
            case kKey_PageDown:
               mark_ok = kTRUE;
               ScreenDown();
               break;
            case kKey_Home:
               mark_ok = kTRUE;
               Home();
               break;
            case kKey_End:
               mark_ok = kTRUE;
               End();
               break;
            case kKey_Insert:           // switch on/off insert mode
               SetInsertMode(GetInsertMode() == kInsert ? kReplace : kInsert);
               break;
            default:
               break;
         }
      }
      if ((event->fState & kKeyShiftMask) && mark_ok) {
         fIsMarked = kTRUE;
         Mark(fCurrent.fX, fCurrent.fY);
         SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED), fWidgetId,
                     kTRUE);
      } else {
         if (fIsMarked) {
            fIsMarked = kFALSE;
            UnMark();
            SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED),
                        fWidgetId, kFALSE);
         }
         fMarkedStart.fY = fMarkedEnd.fY = fCurrent.fY;
         fMarkedStart.fX = fMarkedEnd.fX = fCurrent.fX;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (event->fWindow != fCanvas->GetId())
      return kTRUE;

   if (event->fType == kEnterNotify) {
      if (!fCurBlink)
         fCurBlink = new TViewTimer(this, 500);
      fCurBlink->Reset();
      gSystem->AddTimer(fCurBlink);
   } else {
      if (fCurBlink) fCurBlink->Remove();
      if (fCursorState == 2) {
         DrawCursor(1);
         fCursorState = 1;
      }
   }

   TGTextView::HandleCrossing(event);

   return kTRUE;
}

//______________________________________________________________________________
void TGTextEdit::InsChar(char character)
{
   // Insert a character in the text edit widget.

   char *charstring = 0;
   TGLongPosition pos;

   if (character == '\t') {
      pos.fX = fCurrent.fX;
      pos.fY = fCurrent.fY;
      fText->InsChar(pos, '\t');
      pos.fX++;
      while (pos.fX & 0x7)
         pos.fX++;
      fText->ReTab(pos.fY);
      DrawRegion(0, (Int_t)ToScrYCoord(pos.fY), fCanvas->GetWidth(),
                 UInt_t(ToScrYCoord(pos.fY+1) - ToScrYCoord(pos.fY)));
      SetSBRange(kHorizontal);
      if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t)fCanvas->GetWidth()) {
         if (pos.fX != fText->GetLineLength(fCurrent.fY))
            SetHsbPosition((fVisible.fX+fCanvas->GetWidth()/2)/fScrollVal.fX);
         else
            SetHsbPosition(fVisible.fX/fScrollVal.fX+strlen(charstring));
      }
      SetCurrent(pos);
      return;
#if 0
      Int_t n = Int_t(pos.fX-fCurrent.fX);
      charstring = new char[n+1];
      charstring[n] = '\0';
      for (int j = 0; j < n; j++)
         charstring[j] = ' ';
#endif
   } else {
      fText->InsChar(fCurrent, character);
      pos.fX = fCurrent.fX + 1;
      pos.fY = fCurrent.fY;
      charstring = new char[2];
      charstring[1] = '\0';
      charstring[0] = character;
   }
   SetSBRange(kHorizontal);
   if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t)fCanvas->GetWidth()) {
      if (pos.fX != fText->GetLineLength(fCurrent.fY))
         SetHsbPosition((fVisible.fX+fCanvas->GetWidth()/2)/fScrollVal.fX);
      else
         SetHsbPosition(fVisible.fX/fScrollVal.fX+strlen(charstring));
      if (!fHsb)
         gVirtualX->DrawString(fCanvas->GetId(), fNormGC,
                               (Int_t)ToScrXCoord(fCurrent.fX, fCurrent.fY),
                               Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent),
                               charstring, strlen(charstring));
   } else {
      gVirtualX->CopyArea(fCanvas->GetId(), fCanvas->GetId(), fNormGC,
                          (Int_t)ToScrXCoord(fCurrent.fX, fCurrent.fY),
                          (Int_t)ToScrYCoord(fCurrent.fY), fCanvas->GetWidth(),
                          UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)),
                          (Int_t)ToScrXCoord(pos.fX, fCurrent.fY),
                          (Int_t)ToScrYCoord(fCurrent.fY));
      gVirtualX->ClearArea(fCanvas->GetId(),
                           Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                           Int_t(ToScrYCoord(fCurrent.fY)),
                           UInt_t(ToScrXCoord(fCurrent.fX+strlen(charstring), fCurrent.fY) -
                           ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                           UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
      gVirtualX->DrawString(fCanvas->GetId(), fNormGC,
                            Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                            Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent),
                            charstring, strlen(charstring));
      fCursorState = 2;  // the ClearArea effectively turned off the cursor
   }
   delete [] charstring;
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::DelChar()
{
   // Delete a character from the text edit widget.

   if (fCurrent.fY == 0 && fCurrent.fX == 0) {
      gVirtualX->Bell(0);
      return;
   }

   char *buffer;
   TGLongPosition pos, pos2;
   Long_t len;

   pos.fY = fCurrent.fY;
   pos.fX = fCurrent.fX;

   if (fCurrent.fX > 0) {
      pos.fX--;
      if (fText->GetChar(pos) == 16) {
         do {
            pos.fX++;
            fText->DelChar(pos);
            pos.fX -= 2;
         } while (fText->GetChar(pos) != '\t');
         pos.fX++;
         fText->DelChar(pos);
         pos.fX--;
         fText->ReTab(pos.fY);
         DrawRegion(0, (Int_t)ToScrYCoord(pos.fY), fCanvas->GetWidth(),
                    UInt_t(ToScrYCoord(pos.fY+2)-ToScrYCoord(pos.fY)));
      } else {
         pos.fX = fCurrent.fX;
         fText->DelChar(fCurrent);
         pos.fX = fCurrent.fX - 1;
      }
      if (ToScrXCoord(fCurrent.fX-1, fCurrent.fY) < 0)
         SetHsbPosition((fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
      SetSBRange(kHorizontal);
      DrawRegion(0, (Int_t)ToScrYCoord(pos.fY), fCanvas->GetWidth(),
                 UInt_t(ToScrYCoord(pos.fY+2)-ToScrYCoord(pos.fY)));
   } else {
      if (fCurrent.fY > 0) {
         len = fText->GetLineLength(fCurrent.fY);
         if (len > 0) {
            buffer = fText->GetLine(fCurrent, len);
            pos.fY--;
            pos.fX = fText->GetLineLength(fCurrent.fY-1);
            fText->InsText(pos, buffer);
            pos.fY++;
            delete [] buffer;
         } else
            pos.fX = fText->GetLineLength(fCurrent.fY -1);
         pos2.fY = ToScrYCoord(fCurrent.fY+1);
         pos.fY = fCurrent.fY - 1;
         fText->DelLine(fCurrent.fY);
         len = fText->GetLineLength(fCurrent.fY-1);
         if (ToScrXCoord(pos.fX, fCurrent.fY-1) >= (Int_t)fCanvas->GetWidth())
            SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);

         gVirtualX->CopyArea(fCanvas->GetId(), fCanvas->GetId(), fNormGC, 0,
                             Int_t(pos2.fY), fWidth,
                             UInt_t(fCanvas->GetHeight() - ToScrYCoord(fCurrent.fY)),
                             0, (Int_t)ToScrYCoord(fCurrent.fY));
         if (ToScrYCoord(pos.fY) < 0)
            SetVsbPosition(fVisible.fY/fScrollVal.fY-1);
         DrawRegion(0, (Int_t)ToScrYCoord(pos.fY), fCanvas->GetWidth(),
                    UInt_t(ToScrYCoord(pos.fY+1) - ToScrYCoord(pos.fY)));
         SetSBRange(kVertical);
         SetSBRange(kHorizontal);
      }
   }
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::BreakLine()
{
   // Break a line.

   TGLongPosition pos;
   fText->BreakLine(fCurrent);
   if (ToScrYCoord(fCurrent.fY+2) <= (Int_t)fCanvas->GetHeight()) {
      gVirtualX->CopyArea(fCanvas->GetId(), fCanvas->GetId(), fNormGC, 0,
                          (Int_t)ToScrYCoord(fCurrent.fY+1), fCanvas->GetWidth(),
                          UInt_t(fCanvas->GetHeight()-(ToScrYCoord(fCurrent.fY+2)-
                          ToScrYCoord(fCurrent.fY))),
                          0, (Int_t)ToScrYCoord(fCurrent.fY+2));
      DrawRegion(0, (Int_t)ToScrYCoord(fCurrent.fY), fCanvas->GetWidth(),
                 UInt_t(ToScrYCoord(fCurrent.fY+2) - ToScrYCoord(fCurrent.fY)));

      if (fVisible.fX != 0)
         SetHsbPosition(0);
      SetSBRange(kHorizontal);
      SetSBRange(kVertical);
   } else {
      SetSBRange(kHorizontal);
      SetSBRange(kVertical);
      SetVsbPosition(fVisible.fY/fScrollVal.fY + 1);
      DrawRegion(0, (Int_t)ToScrYCoord(fCurrent.fY),
                 fCanvas->GetWidth(),
                 UInt_t(ToScrYCoord(fCurrent.fY+1) - ToScrYCoord(fCurrent.fY)));
   }
   pos.fY = fCurrent.fY+1;
   pos.fX = 0;
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::ScrollCanvas(Int_t new_top, Int_t direction)
{
   // Scroll the canvas to new_top in the kVertical or kHorizontal direction.

   CursorOff();

   TGTextView::ScrollCanvas(new_top, direction);

   CursorOn();
}

//______________________________________________________________________________
void TGTextEdit::DrawRegion(Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Redraw the text edit widget.

   CursorOff();

   TGTextView::DrawRegion(x, y, width, height);

   CursorOn();
}

//______________________________________________________________________________
void TGTextEdit::PrevChar()
{
   // Go to the previous character.

   if (fCurrent.fY == 0 && fCurrent.fX == 0) {
      gVirtualX->Bell(0);
      return;
   }

   TGLongPosition pos;
   Long_t len;

   pos.fY = fCurrent.fY;
   pos.fX = fCurrent.fX;
   if (fCurrent.fX > 0) {
      pos.fX--;
      while (fText->GetChar(pos) == 16)
         pos.fX--;
      if (ToScrXCoord(pos.fX, pos.fY) < 0) {
         if (fVisible.fX-(Int_t)fCanvas->GetWidth()/2 >= 0)
            SetHsbPosition((fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
         else
            SetHsbPosition(0);
      }
   } else {
      if (fCurrent.fY > 0) {
         pos.fY = fCurrent.fY - 1;
         len = fText->GetLineLength(pos.fY);
         if (ToScrYCoord(fCurrent.fY) <= 0)
            SetVsbPosition(fVisible.fY/fScrollVal.fY-1);
         if (ToScrXCoord(len, pos.fY) >= (Int_t)fCanvas->GetWidth())
            SetHsbPosition((ToScrXCoord(len, pos.fY)+fVisible.fX -
                            fCanvas->GetWidth()/2)/fScrollVal.fX);
         pos.fX = len;
      }
   }
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::NextChar()
{
   // Go to next character.

   Long_t len = fText->GetLineLength(fCurrent.fY);

   if (fCurrent.fY == fText->RowCount()-1 && fCurrent.fX == len) {
      gVirtualX->Bell(0);
      return;
   }

   TGLongPosition pos;
   pos.fY = fCurrent.fY;
   if (fCurrent.fX < len) {
      if (fText->GetChar(fCurrent) == '\t')
         pos.fX = fCurrent.fX + 8 - (fCurrent.fX & 0x7);
      else
         pos.fX = fCurrent.fX + 1;

      if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())
         SetHsbPosition(fVisible.fX/fScrollVal.fX+(fCanvas->GetWidth()/2)/fScrollVal.fX);
   } else {
      if (fCurrent.fY < fText->RowCount()-1) {
         pos.fY = fCurrent.fY + 1;
         if (ToScrYCoord(pos.fY+1) >= (Int_t)fCanvas->GetHeight())
            SetVsbPosition(fVisible.fY/fScrollVal.fY+1);
         SetHsbPosition(0);
         pos.fX = 0;
      }
   }
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::LineUp()
{
   // Make current position first line in window by scrolling up.

   TGLongPosition pos;
   Long_t len;
   if (fCurrent.fY > 0) {
      pos.fY = fCurrent.fY - 1;
      if (ToScrYCoord(fCurrent.fY) <= 0)
         SetVsbPosition(fVisible.fY/fScrollVal.fY-1);
      len = fText->GetLineLength(fCurrent.fY-1);
      if (fCurrent.fX > len) {
         if (ToScrXCoord(len, pos.fY) <= 0) {
            if (ToScrXCoord(len, pos.fY) < 0)
               SetHsbPosition(ToScrXCoord(len, pos.fY)+
                            (fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
            else
               SetHsbPosition(0);
         }
         pos.fX = len;
      } else
         pos.fX = ToObjXCoord(ToScrXCoord(fCurrent.fX, fCurrent.fY)+fVisible.fX, pos.fY);
      while (fText->GetChar(pos) == 16)
         pos.fX++;
      SetCurrent(pos);
   }
}

//______________________________________________________________________________
void TGTextEdit::LineDown()
{
   // Move one line down.

   TGLongPosition pos;
   Long_t len;
   if (fCurrent.fY < fText->RowCount()-1) {
      len = fText->GetLineLength(fCurrent.fY+1);
      pos.fY = fCurrent.fY + 1;
      if (ToScrYCoord(pos.fY+1) > (Int_t)fCanvas->GetHeight())
         SetVsbPosition(fVisible.fY/fScrollVal.fY+1);
      if (fCurrent.fX > len) {
         if (ToScrXCoord(len, pos.fY) <= 0) {
            if (ToScrXCoord(len, pos.fY) < 0)
               SetHsbPosition((ToScrXCoord(len, pos.fY)+fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
            else
               SetHsbPosition(0);
         }
         pos.fX = len;
      } else
         pos.fX = ToObjXCoord(ToScrXCoord(fCurrent.fX, fCurrent.fY)+fVisible.fX, pos.fY);
      while (fText->GetChar(pos) == 16)
         pos.fX++;
      SetCurrent(pos);
   }
}

//______________________________________________________________________________
void TGTextEdit::ScreenUp()
{
   // Move one screen up.

   TGLongPosition pos;
   pos.fX = fCurrent.fX;
   pos.fY = fCurrent.fY - (ToObjYCoord(fCanvas->GetHeight())-ToObjYCoord(0))-1;
   if (fVisible.fY - (Int_t)fCanvas->GetHeight() >= 0) { // +1
      SetVsbPosition((fVisible.fY - fCanvas->GetHeight())/fScrollVal.fY);
   } else {
      pos.fY = 0;
      SetVsbPosition(0);
   }
   while (fText->GetChar(pos) == 16)
      pos.fX++;
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::ScreenDown()
{
   // Move one screen down.

   TGLongPosition pos;
   pos.fX = fCurrent.fX;
   pos.fY = fCurrent.fY + (ToObjYCoord(fCanvas->GetHeight()) - ToObjYCoord(0));
   Long_t count = fText->RowCount()-1;
   if ((Int_t)fCanvas->GetHeight() < ToScrYCoord(count)) {
      SetVsbPosition((fVisible.fY+fCanvas->GetHeight())/fScrollVal.fY);
   } else
      pos.fY = count;
   while (fText->GetChar(pos) == 16)
      pos.fX++;
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::Home()
{
   // Move to beginning of line.

   TGLongPosition pos;
   pos.fY = fCurrent.fY;
   pos.fX = 0;
   SetHsbPosition(0);
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::End()
{
   // Move to end of line.

   TGLongPosition pos;
   pos.fY = fCurrent.fY;
   pos.fX = fText->GetLineLength(pos.fY);
   if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())
      SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
   SetCurrent(pos);
}
