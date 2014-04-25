// @(#)root/utils/src:$Id$
// Author: Philippe Canal November 2011 ;
// 16/04/2010 and Velislava Spasova.
// originated from Zdenek Culik   


/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/rootcint.            *
 *************************************************************************/

#ifndef ROOT__RSCANNER_H__
#define ROOT__RSCANNER_H__

#include <stack>

#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/Type.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "llvm/IR/Module.h"

#include "TMetaUtils.h"

namespace clang {
   class ClassTemplatePartialSpecializationDecl;
   class ClassTemplateDecl;
   class RecordDecl;
   class Stmt;
}

namespace cling {
   class Interpreter;
}

class SelectionRules;

class RScanner: public clang::RecursiveASTVisitor<RScanner>
{
   
public:
   class AnnotatedNamespaceDecl {
   public:
      AnnotatedNamespaceDecl(clang::NamespaceDecl *decl, long index, bool rRequestOnlyTClass) :
      fDecl(decl), fRuleIndex(index), fRequestOnlyTClass(rRequestOnlyTClass){}
      AnnotatedNamespaceDecl() { /* Nothing to do we do not own the pointer; */}
      bool RequestOnlyTClass() const { return fRequestOnlyTClass; }
      const clang::NamespaceDecl* GetNamespaceDecl() const { return fDecl; }
      operator clang::NamespaceDecl const *() const { return fDecl; }
      bool operator<(const AnnotatedNamespaceDecl& right) { return fRuleIndex < right.fRuleIndex; }
   private:
      const clang::NamespaceDecl *fDecl;
      long fRuleIndex;
      bool fRequestOnlyTClass;
   };
   
   typedef std::vector<AnnotatedNamespaceDecl> NamespaceColl_t;   
   typedef std::vector<ROOT::TMetaUtils::AnnotatedRecordDecl>   ClassColl_t;
   typedef std::vector<clang::TypedefNameDecl*> TypedefColl_t;
   typedef void (*DeclCallback)(const char *type);
      
   RScanner (SelectionRules &rules,
             const cling::Interpreter &interpret,
             ROOT::TMetaUtils::TNormalizedCtxt &normCtxt,
             unsigned int verbose = 0);
   
   // Configure the vistitor to also visit template instantiation.
   bool shouldVisitTemplateInstantiations() const { return true; }

   // Don't descend into function bodies.
   bool TraverseStmt(clang::Stmt*) { return true; }

   // Don't descend into templates partial specialization (but only instances thereof).
   bool TraverseClassTemplatePartialSpecializationDecl(clang::ClassTemplatePartialSpecializationDecl*) { return true; }
      
   bool VisitEnumDecl(clang::EnumDecl* D); //Visitor for every EnumDecl i.e. enumeration node in the AST
   bool VisitFieldDecl(clang::FieldDecl* D); //Visitor for e field inside a class
   bool VisitFunctionDecl(clang::FunctionDecl* D); //Visitor for every FunctionDecl i.e. function node in the AST
   bool VisitNamespaceDecl(clang::NamespaceDecl* D); // Visitor for every RecordDecl i.e. class node in the AST
   bool VisitRecordDecl(clang::RecordDecl* D); // Visitor for every RecordDecl i.e. class node in the AST
   bool VisitTypedefNameDecl(clang::TypedefNameDecl* D); // Visitor for every TypedefNameDecl i.e. class node in the AST
   bool VisitVarDecl(clang::VarDecl* D); //Visitor for every VarDecl i.e. variable node in the AST

   bool TreatRecordDeclOrTypedefNameDecl(clang::TypeDecl* typeDecl); //Function called by VisitTypedefNameDecl and VisitRecordDecl
   
   bool TraverseDeclContextHelper(clang::DeclContext *DC); // Here is the code magic :) - every Decl
   // according to its type is processed by the corresponding Visitor method

   // Set a callback to record which are declared.
   DeclCallback SetRecordDeclCallback(DeclCallback callback);

   // Main interface of this class.
   void Scan(const clang::ASTContext &C, bool twoPasses);

   // Utility routines.  Most belongs in TMetaUtils and should be shared with rootcling.cxx
   std::string GetClassName(clang::DeclContext* DC) const;
   void DumpDecl(clang::Decl* D, const char* msg) const;
   bool GetDeclName(clang::Decl* D, std::string& name) const;
   bool GetDeclQualName(clang::Decl* D, std::string& qual_name) const;
   bool GetFunctionPrototype(clang::Decl* D, std::string& prototype) const;

   static const char* fgClangDeclKey; // property key used for CLang declaration objects
   static const char* fgClangFuncKey; // property key for function (demangled) names
   
   // public for now, the list of selected classes.
   ClassColl_t     fSelectedClasses;
   NamespaceColl_t fSelectedNamespaces;
   TypedefColl_t fSelectedTypedefs;
   
   virtual ~ RScanner ();
      
private:

   std::string ConvTemplateArguments(const clang::TemplateArgumentList& list) const;
   std::string ConvTemplateName(clang::TemplateName& N) const;
   std::string ConvTemplateParameterList(clang::TemplateParameterList* list) const;
   std::string ConvTemplateParams(clang::TemplateDecl* D) const;
   void DeclInfo(clang::Decl* D) const;
   std::string ExprToStr(clang::Expr* expr) const;
   std::string FuncParameterList(clang::FunctionDecl* D) const;
   std::string FuncParameters(clang::FunctionDecl* D) const;
   std::string GetClassName(clang::RecordDecl* D) const;
   std::string GetEnumName(clang::EnumDecl* D) const;
   std::string GetLocation(clang::Decl* D) const;
   std::string GetName(clang::Decl* D) const;
   std::string GetSrcLocation(clang::SourceLocation L) const;
   unsigned int FuncModifiers(clang::FunctionDecl* D) const;
   unsigned int fVerboseLevel;
   unsigned int VarModifiers(clang::VarDecl* D) const;
   unsigned int Visibility(clang::Decl* D) const;
   unsigned int VisibilityModifiers(clang::AccessSpecifier access) const;
   void ShowError(const std::string &msg, const std::string &location = "") const;
   void ShowInfo(const std::string &msg, const std::string &location = "") const;
   void ShowTemplateInfo(const std::string &msg, const std::string &location = "") const;
   void ShowWarning(const std::string &msg, const std::string &location = "") const;
   static std::map <clang::Decl*, std::string> fgAnonymousClassMap;
   static std::map <clang::Decl*, std::string> fgAnonymousEnumMap;
   void UnexpectedDecl(clang::Decl* D,const std::string &txt = "") const;
   void UnimplementedDecl(clang::Decl* D,const std::string &txt = "");
   void UnimplementedType(clang::QualType qual_type);
   void UnimplementedType (const clang::Type* T);
   void UnimportantDecl(clang::Decl* D,const std::string &txt = "") const;
   void UnimportantType(clang::QualType qual_type) const;
   void UnknownDecl(clang::Decl* D, const std::string &txt = "") const;
   void UnknownType(clang::QualType qual_type) const;
   void UnsupportedDecl(clang::Decl* D,const std::string &txt = "") const;
   void UnsupportedType(clang::QualType qual_type) const;
   
   const clang::SourceManager* fSourceManager;
   const cling::Interpreter &fInterpreter;
   static const int fgDeclLast = clang::Decl::Var;
   static const int fgTypeLast = clang::Type::TemplateTypeParm;
   bool fDeclTable [ fgDeclLast+1 ];
   clang::Decl * fLastDecl;
   DeclCallback fRecordDeclCallback;
   bool fTypeTable [ fgTypeLast+1 ];
   static int fgAnonymousClassCounter;
   static int fgAnonymousEnumCounter;
   static int fgBadClassCounter;
   ROOT::TMetaUtils::TNormalizedCtxt &fNormCtxt;
   SelectionRules &fSelectionRules;
   std::set<clang::RecordDecl*> fselectedRecordDecls; // Set for O(logN), unordered_set will be better
   bool fFirstPass; // This flag allows to run twice, for example in presence of dict selection and recursive template list manipulations.


};

#endif /* ROOT__RSCANNER_H__ */
